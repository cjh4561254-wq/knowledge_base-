# utils/auth_utils.py
"""
知识库登录鉴权核心层（真实登录鉴权，拦住 chat / import 页面与全部业务接口）
=========================================================================

职责
----
1. 口令散列：标准库 `hashlib.pbkdf2_hmac("sha256", ...)` + `secrets` 随机盐 +
   `hmac.compare_digest` 恒定时间比较。**零新依赖**（venv 里没有 bcrypt/passlib/
   itsdangerous，也不允许装，见下方"为什么不用 bcrypt"）。
2. 用户与会话存储：复用项目已有的 MongoDB（库 kb001），新增两个集合 users / sessions。
3. FastAPI 侧的会话解析依赖：require_user / optional_user / read_session_token。
4. 登录失败限流（按用户名的内存计数器）+ 账号枚举防护。

依赖的既有约定（来自 utils/mongo_history_utils.py，务必保持一致）
---------------------------------------------------------------
* `load_dotenv(override=True)`：不加 override 时进程里可能已存在同名旧变量，
  会读到错的 MONGO_URL（本机实测踩过），所以每次 import 都强制以 .env 覆盖。
* 懒加载单例：模块 import 阶段**绝不**建连。MongoClient 只在第一个真正需要库的
  函数里构造；否则 Mongo 未起时 `import utils.auth_utils` 就会把整个服务搞不起来。
* `serverSelectionTimeoutMS=5000`：pymongo 默认 30000，连不上时每个请求都会把请求
  线程钉死 30 秒。这里沿用项目已实测的 5 秒快速失败思路。

对外契约（web/api/*.py 与 web/page/*.html 按这些名字调用，签名固定）
-----------------------------------------------------------------
常量：SESSION_COOKIE / SESSION_MAX_AGE / REMEMBER_MAX_AGE / LOGIN_PATH
函数：hash_password / verify_password / create_user / get_user_by_username /
      get_user_by_id / authenticate / create_session / get_session /
      touch_session / delete_session / read_session_token / optional_user /
      require_user / is_html_request / auth_status / close
额外便捷函数（不改上面任何签名，只是把 cookie flag 收敛到一处实现）：
      set_session_cookie(response, token, remember) / clear_session_cookie(response)

新增环境变量（只有这一个，主控需要写进 .env）
------------------------------------------
AUTH_COOKIE_SECURE=0    # 1 表示给会话 cookie 加 Secure 标记

典型用法（给并行改 web/* 的同学）
------------------------------
    from fastapi import Depends, Response
    from utils import auth_utils as au

    # 1) 登录接口
    @app.post("/api/login")
    async def login(body: LoginBody, response: Response):
        try:
            user = au.authenticate(body.username, body.password)   # 失败 None
        except au.TooManyLoginAttemptsError as e:
            raise HTTPException(429, detail=str(e))
        except au.AuthError as e:                                    # 含 AuthStorageError
            raise HTTPException(503, detail=str(e))
        if user is None:
            raise HTTPException(401, detail=au.BAD_CREDENTIALS_MESSAGE)  # 统一文案
        token, _session = au.create_session(user["_id"], remember=body.remember)
        au.set_session_cookie(response, token, remember=body.remember)
        return {"ok": True, "user": user}

    # 2) 页面与业务接口：挂依赖即可
    @app.get("/chat.html")
    async def chat_page(user: dict = Depends(au.require_user)): ...

    # 3) 退出
    @app.post("/api/logout")
    async def logout(request: Request, response: Response):
        au.delete_session(au.read_session_token(request))
        au.clear_session_cookie(response)
        return {"ok": True}

    # 4) 建第一个用户（库里 0 用户时 require_user 会把所有人挡在门外，需要一个入口）
    au.create_user("admin", "xxxxxxxx", "管理员")
"""

import base64
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from fastapi import HTTPException, Request, Response
from pymongo import ASCENDING, MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError, OperationFailure, PyMongoError

from tool.logger import logger

# 与 utils/mongo_history_utils.py 完全一致：必须 override=True，
# 否则外壳环境里残留的同名变量会压过 .env（本机实测过这个坑）
load_dotenv(override=True)

# ====================
# 公开常量（契约的一部分，别改值语义）
# ====================

#: 会话 cookie 名（前端不需要读它：HttpOnly 读不到，纯服务端使用）
SESSION_COOKIE: str = "kb_session"

#: 会话型 cookie 的有效期（秒）≈ 12 小时
#: 为什么取 12h：工作日一天的连续使用不必重新登录，同时偷到的 cookie 有天然上限
SESSION_MAX_AGE: int = 12 * 60 * 60  # 43200

#: 勾选"记住我"时的有效期（秒）≈ 30 天
REMEMBER_MAX_AGE: int = 30 * 24 * 60 * 60  # 2592000

#: 未登录时把 HTML 导航请求重定向到这里（由并行改 web/* 的同学提供这个静态页）
LOGIN_PATH: str = "/login.html"

#: 登录失败的统一文案。
#: 为什么必须是一句：任何"用户不存在"/"密码错误"级别的差别都等于把有效用户名清单
#: 白送给枚举者（本项目的核心防护点之一）。调用方请直接复用这个常量。
BAD_CREDENTIALS_MESSAGE: str = "用户名或密码错误"

#: 集合名（与库里既有的 chat_message 平级，互不影响）
USERS_COLLECTION: str = "users"
SESSIONS_COLLECTION: str = "sessions"

# ====================
# 口令散列参数
# ====================

_HASH_ALGORITHM = "pbkdf2_sha256"

#: PBKDF2-HMAC-SHA256 迭代次数。
#: **不是拍脑袋定的**：本机（Python 3.12.8 / Windows / 无 GPU 参与）用
#: hashlib.pbkdf2_hmac 实测单次散列耗时（time.perf_counter，各取 3 次中位数）：
#:     100_000 -> 29.5ms     200_000 -> 57.5ms     400_000 -> 118.0ms
#:     700_000 -> 203.2ms    850_000 -> 253.6ms  1_000_000 -> 293.5ms
#:     1_200_000 -> 348.6ms  1_500_000 -> 505.3ms
#: 需求是"单次 200–500ms 打底"，取 1_000_000：本机 294ms，正部落在区间中部，
#: 即使跑在约 1.5 倍慢的生产机上（≈440ms）仍不破 500ms 上限；而 1_500_000 已经
#: 踩线（本机 505ms），慢机器会直接超预算，故不取。
_PBKDF2_ITERATIONS = 1_000_000

#: 迭代次数上限：库里/备份被篡改时，攻击者可以塞一个"迭代数=10 亿"的假文档，
#: 让服务端在校验时把 CPU 打满（DoS）。超过上限的散列串直接判为无效（不计算）。
_PBKDF2_ITERATIONS_MAX = 3_000_000

_SALT_BYTES = 16  # 128bit 随机盐：唯一且足够长，碰撞概率可忽略
_HASH_BYTES = 32  # sha256 输出 256bit
_FIELD_SEP = "$"

#: 用户名规则（归一化之后再校验）：长度 3–32，仅允许中文/字母/数字/._-
_USERNAME_MIN_LEN = 3
_USERNAME_MAX_LEN = 32
_USERNAME_RE = re.compile(r"^[0-9a-z_\.\-\u4e00-\u9fff]+$")

#: 口令长度规则（8–128）
_PASSWORD_MIN_LEN = 8
_PASSWORD_MAX_LEN = 128

# ====================
# 登录失败限流参数（内存，重启清零）
# ====================

_LOGIN_MAX_FAILURES = 5  # 同一用户名窗口内最多 5 次失败
_LOGIN_WINDOW_SECONDS = 15 * 60  # 15 分钟
#: _login_failures 的容量上限：防止有人拿大量随机用户名把内存刷爆（用户名枚举本身就是攻击面）
_LOGIN_TRACKER_MAX_KEYS = 10000

#: touch_session 的写库节流：同一 token 60 秒内最多评估/写一次
_TOUCH_THROTTLE_SECONDS = 60.0
_TOUCH_CACHE_MAX_KEYS = 8192

#: session token 最大可接受长度（token_urlsafe(32) 产出 43 字符，留足余量）
_TOKEN_MAX_LEN = 128

# ====================
# 异常体系
# ====================


class AuthError(Exception):
    """鉴权相关错误的基类：detail 文案面向用户，可以直接塞进 HTTPException。"""


class InvalidUsernameError(AuthError):
    """用户名不合法（长度/字符集不满足规则）。"""


class InvalidPasswordError(AuthError):
    """口令不合规（长度不在 8–128）。"""


class UsernameTakenError(AuthError):
    """用户名已存在（create_user 撞上 users.username 唯一索引）。"""


class TooManyLoginAttemptsError(AuthError):
    """登录失败次数超限，该用户名暂时被锁定（authenticate 抛出，非返回 None）。"""


class AuthStorageError(AuthError):
    """users/sessions 所在 MongoDB 不可用（连不上/超时/写失败）。

    为什么要有专门的异常类型：鉴权存储挂掉时**绝不能**退化成"当作未登录"，
    那样会把数据库故障伪装成用户故障（所有人突然登不上却看不到原因）；
    也不能放任 pymongo 的 30 秒选主超时冒泡。这里统一成一条可读的 503 文案。
    """


# ====================
# Mongo 连接：函数内延迟建连 + 幂等建索引 + 快速失败
# ====================

#: serverSelectionTimeoutMS 沿用 .env 的 MONGO_SERVER_SELECTION_TIMEOUT_MS（默认 5000ms）
_SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000"))

_storage_lock = threading.Lock()
_client: Optional[MongoClient] = None
_db = None
_indexes_ready = False
_last_index_attempt: float = 0.0
#: 建索引失败后的重试冷却（秒）：Mongo 挂着时不让每个请求都再撞一遍 5 秒超时
_INDEX_RETRY_COOLDOWN_SECONDS = 30.0


def _mongo_url() -> str:
    """读 Mongo 连接串。兜底默认值与 utils/mongo_history_utils.py 保持一致。"""
    return (os.getenv("MONGO_URL") or "mongodb://localhost:27017").strip()


def _mongo_db_name() -> str:
    return (os.getenv("MONGO_DB_NAME") or "kb001").strip() or "kb001"


def _cookie_secure() -> bool:
    """会话 cookie 是否加 Secure。

    为什么默认关：本机全部走 http://127.0.0.1，浏览器在明文 HTTP 上会**丢弃**带
    Secure 的 cookie，一开就变成"永远登不上"。上生产（HTTPS）时必须设
    AUTH_COOKIE_SECURE=1，否则会话 cookie 明文可被链路中间人读取。
    """
    return (os.getenv("AUTH_COOKIE_SECURE") or "0").strip().lower() in ("1", "true", "yes", "on")


def _utcnow() -> datetime:
    """统一时间源：带时区的 UTC now。

    为什么不用 datetime.now()：服务器本地时区/夏令时一变，会话有效期就跟着漂
    （历史上"提前失效"和"多活 8 小时"两类事故都来自这里）。库里一律存 UTC，
    比较也全在 UTC 做，跟部署机时区彻底解耦。
    """
    return datetime.now(timezone.utc)


def _as_utc(value: Any) -> Optional[datetime]:
    """把库里的时间字段规整成带时区 UTC datetime；不是 datetime 返回 None。

    为什么要单独处理：BSON 日期没有时区槽位，pymongo 默认返回 **naive** datetime
    （值本身是 UTC）。直接拿 naive 去和 aware 的 _utcnow() 比较会 TypeError，
    所以这里显式补 UTC tzinfo，而不是在客户端开 tz_aware（后者会顺带改变
    chat_message 等既有集合的读取形态，属于不必要的横向影响）。
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    # 兼容极少数被写成 ISO 字符串的脏数据
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _iso(dt: Any) -> str:
    """datetime -> ISO8601 字符串（JSON 可序列化，统一带 +00:00）。"""
    aware = _as_utc(dt)
    return aware.isoformat() if aware is not None else ""


def _create_index_idempotent(collection, keys, name: str, **kwargs) -> None:
    """建索引，且"已存在"绝不算失败。

    为什么需要包装：pymongo 的 create_index 只在**key 与选项完全一致**时幂等；
    同名不同定义会抛 OperationFailure(IndexExists 85 / IndexOptionsConflict 86)，
    异名同 key 也会抛。这些情况说明库里已经有可用索引，重启时不该让服务起不来，
    所以按错误码/文案判定后降级为告警；其它异常（权限、 Mongo 挂了）原样上抛。
    """
    try:
        collection.create_index(keys, name=name, **kwargs)
    except OperationFailure as e:
        code = getattr(e, "code", None)
        msg = str(e)
        # 85 = IndexExists（同名同定义重复创建）；86 = IndexOptionsConflict（同名不同定义）
        if code in (85, 86) or "already exists" in msg.lower():
            logger.warning(
                f"[auth] 索引 {name} 已存在（code={code}），沿用现有定义，不重复创建。"
                f"若该字段尚未按预期建 TTL/唯一索引，请人工核对：{msg}"
            )
            return
        raise
    except TypeError as e:
        # 老版本 pymongo 对不支持的 kwargs 抛 TypeError，明确暴露而不是静默吞掉
        raise AuthStorageError(f"创建索引 {name} 的参数不被当前 pymongo 支持：{e}") from e


def _ensure_indexes(db) -> None:
    """建（并保证幂等）users/sessions 需要的索引。

    * users.username 唯一：重名注册在**数据库层**被挡住，避免"先查再插"的竞态窗口。
    * sessions.token 唯一：token 主键化，get/delete 走唯一索引，等值命中。
    * sessions.expires_at TTL(expireAfterSeconds=0)：到点由 Mongo 自己回收，
      省掉一个清理任务。为什么代码里还要再判 expires_at：TTL 后台线程约每 60 秒
      才扫一轮，最长有 1 分钟的"已过期但还在库里"的窗口，安全判定必须以代码为准。
    * sessions.user_id 普通索引：将来"改密码后踢掉该用户全部会话"按此字段批删。
    """
    users = db[USERS_COLLECTION]
    sessions = db[SESSIONS_COLLECTION]
    _create_index_idempotent(
        users, [("username", ASCENDING)], "uniq_users_username", unique=True, background=True
    )
    _create_index_idempotent(
        sessions, [("token", ASCENDING)], "uniq_sessions_token", unique=True, background=True
    )
    _create_index_idempotent(
        sessions,
        [("expires_at", ASCENDING)],
        "idx_sessions_expires_at",
        expireAfterSeconds=0,
        background=True,
    )
    _create_index_idempotent(
        sessions, [("user_id", ASCENDING)], "idx_sessions_user_id", background=True
    )


def _get_db():
    """返回 Mongo Database 对象；**首次真正用到时才建连并建索引**。

    失败统一抛 AuthStorageError（带可读文案），且：
    * 网络类失败受 serverSelectionTimeoutMS=5000 约束 → 最多 5 秒，绝不 30 秒假死；
    * 建索引失败后进入 30 秒冷却，Mongo 持续不可达时不会每个请求都重撞一遍超时；
    * MongoClient 只构造一次（含失败那次）→ 不会每次请求都新起一批拓扑监控线程。
    """
    global _client, _db, _indexes_ready, _last_index_attempt

    # 快路径：已经就绪（绝大多数请求只走这里，无锁）
    if _db is not None and _indexes_ready:
        return _db

    with _storage_lock:
        # 双检：等锁期间可能已有别的线程建好了
        if _db is not None and _indexes_ready:
            return _db

        # 在锁内统一取一次连接串：既给 MongoClient 用，也给下面的报错文案用
        # （若只在 `if _client is None` 分支里赋值，第二次进入本函数时文案里的 url 会未定义）
        url = _mongo_url()
        if _client is None:
            try:
                _client = MongoClient(
                    url,
                    # 三个超时配套压到 5 秒量级：选主 / TCP 建连 / 单次 socket 读写
                    serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS,
                    connectTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS,
                    socketTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS,
                    # 本服务是单进程（内存任务表决定了不能开多 worker），20 条连接够用
                    maxPoolSize=20,
                )
            except Exception as e:  # 连接串非法（ValueError）等构造期错误
                _client = None
                raise AuthStorageError(f"MongoDB 客户端创建失败（请检查 MONGO_URL）：{e}") from e

        if _db is None:
            _db = _client[_mongo_db_name()]

        if not _indexes_ready:
            now = time.monotonic()
            in_cooldown = (
                _last_index_attempt > 0.0
                and (now - _last_index_attempt) < _INDEX_RETRY_COOLDOWN_SECONDS
            )
            if in_cooldown:
                # 冷却期内不再重撞超时：接受"索引可能缺失"的降级继续跑。
                # 代价：唯一索引万一还没建成，重名注册有小概率漏过去 —— 比整站卡死好。
                logger.warning(
                    "[auth] 索引创建仍在冷却期内，本次跳过重试（Mongo 可能未就绪）。"
                )
            else:
                _last_index_attempt = now
                try:
                    _ensure_indexes(_db)
                    _indexes_ready = True
                    logger.info(
                        f"[auth] MongoDB 已连接并确认鉴权索引：db={_mongo_db_name()} "
                        f"collections={USERS_COLLECTION},{SESSIONS_COLLECTION}"
                    )
                except (ConnectionFailure, PyMongoError) as e:
                    # 明确降级：给出"为什么"和"下一步查什么"，而不是堆栈或 30 秒假死
                    raise AuthStorageError(
                        f"登录存储暂不可用（MongoDB {url} 的鉴权索引初始化失败）：{e}；"
                        f"请确认 Mongo 已启动且 MONGO_URL/MONGO_DB_NAME 正确"
                    ) from e
                except Exception as e:
                    raise AuthStorageError(f"登录存储初始化失败：{e}") from e
    return _db


def close() -> None:
    """释放 MongoClient（给测试脚本与进程退出用）。

    行为：关闭连接池并把单例与"索引已就绪"标记全部复位，close() 之后再调用业务函数
    会按**当前**环境变量重新建连 —— 这正是自检脚本能用假 MONGO_URL 验证降级路径的原因。
    重复调用安全（幂等）。
    """
    global _client, _db, _indexes_ready, _last_index_attempt
    with _storage_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception as e:  # 关闭失败无关紧要，但要知道
                logger.warning(f"[auth] MongoClient 关闭异常（忽略）：{e}")
        _client = None
        _db = None
        _indexes_ready = False
        _last_index_attempt = 0.0
    logger.info("[auth] MongoClient 已释放，下次访问会按当前环境变量重连")


# ====================
# 口令散列与校验
# ====================


def _b64e(raw: bytes) -> str:
    """url-safe base64 且去掉 '='：散列串要按 '$' 切 4 段，不能混进分隔符。"""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> Optional[bytes]:
    """逆 _b64e，补回 base64 缺失的 padding；非法字符返回 None（不抛）。"""
    if not isinstance(text, str) or not text:
        return None
    try:
        pad = "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text + pad)
    except (ValueError, TypeError):
        return None


def hash_password(password: str) -> str:
    """生成口令散列串，格式：``pbkdf2_sha256$<iterations>$<saltB64>$<hashB64>``。

    为什么格式里要带算法名和迭代数（而不是只存一个哈希）：
    口令散列的强度是要随算力进步的"可调参数"。把算法与迭代数写进串里，以后
    把 PBKDF2 换成 argon2、或把 1_000_000 提到 2_000_000 时，**老用户的老散列仍然
    能被正确校验**（verify_password 按串内的参数算），新写入的用新参数，
    可以在用户"改密码/登录成功"时顺手升级，完全不需要全量重置密码。
    将来真要换 argon2，只需改 hash_password + verify_password 两个函数，
    存储格式（4 段、'$' 分隔）与所有调用方都不用动。

    盐由 secrets（CSPRNG）每次新生成：同一密码两次散列结果不同，彩虹表失效，
    也让"两个用户密码相同"这件事不泄露。
    """
    if not isinstance(password, str):
        raise InvalidPasswordError("口令必须是字符串")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        _HASH_ALGORITHM.split("_", 1)[1], password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return _FIELD_SEP.join(
        [_HASH_ALGORITHM, str(_PBKDF2_ITERATIONS), _b64e(salt), _b64e(digest)]
    )


def verify_password(password: str, stored: str) -> bool:
    """校验口令。任何格式问题都只返回 False，**绝不抛异常**。

    为什么：调用点（登录接口）在异常路径上很容易把内部结构写进响应/日志；
    库里存了脏数据（手改文档、迁移截断）时，正确行为是"这条散列校验不过"，
    而不是把 500 甩给用户。

    为什么用 hmac.compare_digest：内置 `==` 在第一个不匹配字节就返回，
    攻击者可以靠响应时长的微小差异逐字节猜出散列（计时侧信道）。
    恒定时间比较把这个侧信道关掉。
    """
    if not isinstance(password, str) or not isinstance(stored, str):
        return False
    parts = stored.split(_FIELD_SEP)
    if len(parts) != 4:
        return False
    algorithm, raw_iters, raw_salt, raw_hash = parts
    # 只认我们自己写的算法名；未知/历史格式一律 False（将来在此处加分支即可）
    if algorithm != _HASH_ALGORITHM:
        return False
    try:
        iterations = int(raw_iters)
    except (TypeError, ValueError):
        return False
    if iterations <= 0 or iterations > _PBKDF2_ITERATIONS_MAX:
        # 上限拦截：见 _PBKDF2_ITERATIONS_MAX 的 DoS 说明
        return False
    salt = _b64d(raw_salt)
    expected = _b64d(raw_hash)
    if salt is None or expected is None:
        return False
    try:
        digest = hashlib.pbkdf2_hmac(
            algorithm.split("_", 1)[1], password.encode("utf-8"), salt, iterations
        )
    except (TypeError, ValueError):
        # 极端脏数据（例如 salt 解出 0 字节）也不外抛
        return False
    return hmac.compare_digest(digest, expected)


#: 假盐（固定 ASCII 字节）：用于"用户不存在"路径的等效耗时填充
_DUMMY_SALT = b"auth_utils_dummy_salt"
#: 假哈希：**懒算**（首次用到时才算一次）。
#: 为什么不在模块顶层直接算出来：那样每个进程 import 本模块都要白烧 294ms CPU，
#: 而绝大多数进程（比如只走 require_user 的请求）永远用不到这条路径。
#: 为什么派生值用固定 ASCII 文本而不是随机数或项目里中文字符串：日志/源码文件的
#: 编码差异（GBK / UTF-8 BOM）会让派生值随环境变化，一旦变化，PBKDF2 的耗时也可能
#: 跟着抖动，反而把刚消掉的"用户是否存在"计时侧信道重新引入。
_DUMMY_HASH: Optional[bytes] = None
_dummy_hash_lock = threading.Lock()


def _dummy_hash_bytes() -> bytes:
    """假哈希（必然与任何真实用户的散列不相等，所以 _dummy_verify 必然返回 False）。"""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        with _dummy_hash_lock:
            if _DUMMY_HASH is None:
                _DUMMY_HASH = hashlib.pbkdf2_hmac(
                    _HASH_ALGORITHM.split("_", 1)[1],
                    b"auth_utils_timing_equalizer",
                    _DUMMY_SALT,
                    _PBKDF2_ITERATIONS,
                )
    return _DUMMY_HASH


def _dummy_verify(password: str) -> bool:
    """对固定假盐跑一次**等量**的 PBKDF2，并与一个必然不相等的假哈希恒定时间比较。

    为什么要花这 294ms：如果"用户不存在"直接 return False，它比"密码错误"快约
    300ms（后者必须算一次 PBKDF2）。攻击者不需要猜到密码，只要拿一堆用户名
    计时，慢的那些就是真实账号 —— 计时差把"账号枚举"变成了免费午餐。
    补一次同参数量级的计算后，两条路径耗时基本一致（自检里实测两者差 <10%）。
    假盐 21 字节、真盐 16 字节：HMAC-SHA256 的分组是 64 字节，两者都只需一次 pad，
    所以盐长差异不影响耗时，不需要为对齐成本再截盐。
    """
    try:
        digest = hashlib.pbkdf2_hmac(
            _HASH_ALGORITHM.split("_", 1)[1],
            (password or "").encode("utf-8"),
            _DUMMY_SALT,
            _PBKDF2_ITERATIONS,
        )
        return hmac.compare_digest(digest, _dummy_hash_bytes())
    except Exception:
        return False


# ====================
# 用户名归一化 / 校验
# ====================


def normalize_username(raw: Optional[str]) -> str:
    """用户名归一化：去掉所有空白（含全角空格）→ 转小写 → 去首尾点/横线。

    为什么要归一化："Admin@X" 与 "admin@x"、" 管理员 " 与 "管理员" 在人的直觉里
    是同一个账号。不归一化就会留下"同一个人注册两个名"的空隙，
    也让限流计数被大小写劈成两半（各 5 次 = 实际 10 次尝试）。

    去掉首尾 '.' / '-'：邮箱式用户名常带点分隔，大小写又无关，
    归一化后 "a.b" 与 "a-b" 仍是不同账号，但 ".a." 这种装饰名不再出现。
    """
    if not isinstance(raw, str):
        return ""
    text = "".join(raw.split())  # 覆盖空格/制表/换行/\u3000（str.split() 按 Unicode 空白切）
    return text.lower().strip("._-")


def _validate_username(username: str) -> str:
    """归一化 + 校验，返回规范化后的用户名；不合法抛 InvalidUsernameError。"""
    normalized = normalize_username(username)
    if not normalized:
        raise InvalidUsernameError("用户名不能为空")
    # 归一化后才量长度：否则 "ab"（含空格的前后填充）这种会因空格被算进长度而误判
    if len(normalized) < _USERNAME_MIN_LEN:
        raise InvalidUsernameError(
            f"用户名太短，至少 {_USERNAME_MIN_LEN} 个字符（归一化后为「{normalized}」）"
        )
    if len(normalized) > _USERNAME_MAX_LEN:
        raise InvalidUsernameError(f"用户名太长，最多 {_USERNAME_MAX_LEN} 个字符")
    if _USERNAME_RE.fullmatch(normalized) is None:
        raise InvalidUsernameError(
            "用户名只能包含中文、字母、数字和 . _ -，且不能含空格或其它符号"
        )
    return normalized


def _validate_password(password: Any) -> str:
    """口令校验（长度 8–128）。

    为什么上限 128：PBKDF2 对超长输入是纯 CPU 放大，128 足够人类使用，
    又能挡住"一次 POST 传 1MB 密码"的廉价 DoS。
    """
    if not isinstance(password, str):
        raise InvalidPasswordError("密码必须是字符串")
    length = len(password)
    if length < _PASSWORD_MIN_LEN or length > _PASSWORD_MAX_LEN:
        raise InvalidPasswordError(
            f"密码长度需在 {_PASSWORD_MIN_LEN}–{_PASSWORD_MAX_LEN} 个字符之间（当前 {length}）"
        )
    return password


# ====================
# 文档脱敏：出参绝不带口令散列
# ====================


def _public_user(doc: Optional[dict]) -> Optional[dict]:
    """把 users 文档转成可对外的 user dict（_id 转字符串，散列字段全部剥掉）。

    为什么单独抽一个函数、并且每个读路径都过一遍：只要有一次把原始 doc 直接
    return 给前端，pbkdf2 串就漏了（离线爆破只等一台机器算）。集中剥字段比
    指望每个调用方记得 pop 更可靠。
    """
    if not doc:
        return None
    out = dict(doc)
    # 防御性剥字段：password_hash 是当前字段名，同时兜住历史命名/同义词
    for secret_key in ("password_hash", "password", "pwd", "salt", "hash", "pwd_hash"):
        out.pop(secret_key, None)
    oid = out.get("_id")
    out["_id"] = str(oid) if oid is not None else ""
    if isinstance(out.get("created_at"), datetime) or isinstance(out.get("created_at"), str):
        out["created_at"] = _iso(out.get("created_at"))
    last_login = out.get("last_login_at")
    if last_login is not None:
        out["last_login_at"] = _iso(last_login)
    return out


def _public_session(doc: Optional[dict]) -> Optional[dict]:
    """会话文档对外形态：_id 转字符串、**不回传 token**。

    为什么不回传 token：cookie 是 HttpOnly，前端读不到是设计而非缺陷；
    如果服务端把 token 抄一份塞进 JSON，前端就可能把它存进 localStorage，
    于是"只有浏览器传输层能碰到的凭据"升级成"任何 XSS 脚本都能读走的字符串"。
    """
    if not doc:
        return None
    out = dict(doc)
    out.pop("token", None)  # 关键：token 只存在于 cookie 里
    oid = out.get("_id")
    out["session_id"] = str(oid) if oid is not None else ""
    out.pop("_id", None)
    out["user_id"] = str(out.get("user_id") or "")
    for key in ("created_at", "last_seen_at", "expires_at"):
        if key in out:
            out[key] = _iso(out.get(key))
    return out


def _to_object_id(raw: Any) -> Optional[ObjectId]:
    """宽松地把 user id 转 ObjectId；非法返回 None（不抛）。"""
    if isinstance(raw, ObjectId):
        return raw
    try:
        return ObjectId(str(raw))
    except (InvalidId, TypeError, ValueError):
        return None


# ====================
# 用户：创建与查询
# ====================


def create_user(username: str, password: str, display_name: str = "") -> dict:
    """创建用户，返回**不含密码散列**的 user 文档（含字符串 _id / username / display_name …）。

    重名 → 抛 UsernameTakenError（AuthError 子类）。
    用户名/口令不合规则 → InvalidUsernameError / InvalidPasswordError。
    Mongo 不可用 → AuthStorageError。

    为什么用"直接 insert 并捕获 DuplicateKeyError"，而不是先 find_one 查重：
    check-then-insert 之间有竞态窗口，并发注册可以把同名用户写进库（唯一索引
    此时是最后一道防线，绝不能只靠应用层判断）。撞上唯一索引就翻译成人话文案。
    """
    normalized = _validate_username(username)
    _validate_password(password)
    name = (display_name or "").strip()[:64]

    doc = {
        "username": normalized,
        "password_hash": hash_password(password),
        "display_name": name,
        "created_at": _utcnow(),
        "last_login_at": None,  # 登录成功时更新，便于发现"僵尸账号"
        "is_active": True,  # 预留禁用位（禁用比删记录更可审计，本层先落字段）
    }
    try:
        db = _get_db()
        db[USERS_COLLECTION].insert_one(doc)
    except DuplicateKeyError as e:
        # 文案不带任何"已存在账号的其它信息"（例如 display_name），
        # 否则注册接口就成了账号信息探测接口
        raise UsernameTakenError(f"用户名「{normalized}」已被占用，请换一个") from e
    except AuthError:
        raise
    except PyMongoError as e:
        raise AuthStorageError(f"创建用户失败（登录存储异常）：{e}") from e

    logger.info(f"[auth] 新建用户：username={normalized} display_name={name or '-'}")
    return _public_user(doc) or {}


def get_user_by_username(username: str) -> dict | None:
    """按（会自动归一化的）用户名查用户，返回不含散列的 user dict；不存在返回 None。"""
    normalized = normalize_username(username)
    if not normalized:
        return None
    try:
        db = _get_db()
        doc = db[USERS_COLLECTION].find_one({"username": normalized})
    except AuthError:
        raise
    except PyMongoError as e:
        raise AuthStorageError(f"读取用户失败（登录存储异常）：{e}") from e
    return _public_user(doc)


def get_user_by_id(user_id: str) -> dict | None:
    """按 _id 查用户（sessions.user_id 存的就是这个 ObjectId），不存在返回 None。"""
    oid = _to_object_id(user_id)
    if oid is None:
        return None
    try:
        db = _get_db()
        doc = db[USERS_COLLECTION].find_one({"_id": oid})
    except AuthError:
        raise
    except PyMongoError as e:
        raise AuthStorageError(f"读取用户失败（登录存储异常）：{e}") from e
    return _public_user(doc)


# ====================
# 登录失败限流（内存，线程安全，重启清零）
# ====================

_login_fail_lock = threading.Lock()
#: username -> 失败时刻（time.monotonic 秒）列表
_login_failures: dict[str, list[float]] = {}


def _failure_count(username_key: str, now_mono: float) -> Tuple[int, float]:
    """清理窗口外的旧记录后返回 (窗口内失败次数, 最早一次失败的年龄)。

    调用方必须已持有 _login_fail_lock。
    """
    stamps = _login_failures.get(username_key)
    if not stamps:
        return 0, 0.0
    fresh = [ts for ts in stamps if now_mono - ts < _LOGIN_WINDOW_SECONDS]
    if fresh:
        _login_failures[username_key] = fresh
    else:
        _login_failures.pop(username_key, None)
    # 年龄取"窗口内最早一次"：用来算还需等多久才能再试
    return len(fresh), (now_mono - min(fresh)) if fresh else 0.0


def _throttle_wait_seconds(username_key: str) -> float:
    """若该用户名已被限流，返回还需等待的秒数；未限流返回 0。"""
    with _login_fail_lock:
        count, _age = _failure_count(username_key, time.monotonic())
        if count < _LOGIN_MAX_FAILURES:
            return 0.0
        oldest = min(_login_failures[username_key])
        return max(0.0, _LOGIN_WINDOW_SECONDS - (time.monotonic() - oldest))


def _record_login_failure(username_key: str) -> None:
    """累加一次失败。

    为什么真实用户与不存在的用户都要记：限流的意义是挡"猜口令"，
    而只给真实用户计数就等于告诉攻击者"随便拿不存在的名可以无限试"。
    """
    now_mono = time.monotonic()
    with _login_fail_lock:
        # 顺手做容量治理：键数超限时先清掉窗口内已无失败记录的项，
        # 仍超限就整体清空（宁可放宽一次限流，也不让内存被随机用户名撑爆）
        if len(_login_failures) > _LOGIN_TRACKER_MAX_KEYS:
            for key in [k for k, v in _login_failures.items() if not v]:
                _login_failures.pop(key, None)
            if len(_login_failures) > _LOGIN_TRACKER_MAX_KEYS:
                logger.warning("[auth] 登录限流表过大，已整体清空")
                _login_failures.clear()
        stamps = [ts for ts in _login_failures.get(username_key, []) if now_mono - ts < _LOGIN_WINDOW_SECONDS]
        stamps.append(now_mono)
        _login_failures[username_key] = stamps


def _clear_login_failures(username_key: str) -> None:
    """登录成功后清零（避免历史失败把下一次正常登录误伤）。"""
    with _login_fail_lock:
        _login_failures.pop(username_key, None)


def login_throttle_remaining(username: str) -> float:
    """给登录页用的只读探针：该用户名还需等多少秒（0 = 可以试）。"""
    return _throttle_wait_seconds(normalize_username(username))


def authenticate(username: str, password: str) -> dict | None:
    """校验用户名+口令；成功返回不含散列的 user dict，失败返回 **None**。

    两个例外（都是主动抛，调用方应该 try/except AuthError）：
    * TooManyLoginAttemptsError：命中限流，文案可读（"…请 N 分钟后再试"）。
    * AuthStorageError：Mongo 连不上 —— 存储故障不能伪装成"密码错"。
    其它一切失败（用户不存在 / 口令错 / 参数不合规则 / 账号被禁用）都是 None，
    而且**返回的 None 无法区分**，这是防枚举的关键。

    为什么先做限流再查库：被限流时不该继续消耗 PBKDF2 的 294ms CPU 与一次 Mongo 往返。
    """
    normalized = _validate_username(username)
    if not isinstance(password, str) or not password:
        # 空口令直接拒：不去查库，也就不给"用户名是否存在"任何可观测信号
        return None

    wait = _throttle_wait_seconds(normalized)
    if wait > 0:
        minutes = max(1, int(round(wait / 60.0)))
        raise TooManyLoginAttemptsError(
            f"登录失败次数过多，请 {minutes} 分钟后再试"
        )

    try:
        db = _get_db()
        doc = db[USERS_COLLECTION].find_one({"username": normalized})
    except AuthError:
        raise
    except PyMongoError as e:
        raise AuthStorageError(f"登录校验失败（登录存储异常）：{e}") from e

    if doc is None:
        # 用户不存在：跑一次等效 PBKDF2 抹平耗时差（见 _dummy_verify 的说明），
        # 并且同样计入失败次数 —— 不存在的用户名也别变成"免费的无限试"
        _dummy_verify(password)
        _record_login_failure(normalized)
        logger.warning(f"[auth] 登录失败（用户名或密码错误）：username={normalized}")
        return None

    if doc.get("is_active") is False:
        # 禁用账号也走一次等效计算：避免"禁用状态"成为新的可观测差异
        _dummy_verify(password)
        _record_login_failure(normalized)
        logger.warning(f"[auth] 登录被拒（账号已禁用）：username={normalized}")
        return None

    if not verify_password(password, doc.get("password_hash", "")):
        _record_login_failure(normalized)
        # 这一句刻意用与"用户不存在"完全相同的文案：日志层面也不区分，
        # 否则能读到日志（或指标看板）的人同样能枚举账号
        logger.warning(f"[auth] 登录失败（用户名或密码错误）：username={normalized}")
        return None

    _clear_login_failures(normalized)
    now = _utcnow()
    try:
        db[USERS_COLLECTION].update_one({"_id": doc["_id"]}, {"$set": {"last_login_at": now}})
        doc["last_login_at"] = now
    except PyMongoError as e:
        # 登录已经成功，last_login_at 只是审计字段：写失败不该把用户推出门外
        logger.warning(f"[auth] 更新 last_login_at 失败（不影响登录）：{e}")

    # 绝不打印 password / password_hash：见本文件顶部的日志纪律
    logger.info(f"[auth] 登录成功：username={normalized}")
    return _public_user(doc)


# ====================
# 会话：创建 / 校验 / 续期 / 注销
# ====================


def create_session(user_id: str, remember: bool = False) -> tuple[str, dict]:
    """为已登录用户建会话，返回 (token, session_doc)。

    token 用 secrets.token_urlsafe(32)：32 字节 CSPRNG 熵（256bit），
    可枚举空间是 2**256 量级，"猜一个有效会话"在物理上不可行；
    也刻意不用 uuid4（uuid4 虽随机但只 122bit 熵）或自增 id（直接可枚举）。

    session_doc 是**不含 token** 的对外形态；token 只应由调用方写进 cookie。
    """
    oid = _to_object_id(user_id)
    if oid is None:
        raise AuthError("创建会话失败：user_id 不是合法的 ObjectId")
    # 先确认用户真的存在：给不存在（或已删除）的用户建会话，
    # 会产生一条永远无法通过 _resolve_session 校验、却要等 TTL 才被回收的孤儿记录
    if get_user_by_id(str(oid)) is None:
        raise AuthError("创建会话失败：用户不存在")

    max_age = REMEMBER_MAX_AGE if remember else SESSION_MAX_AGE
    now = _utcnow()
    token = secrets.token_urlsafe(32)
    doc = {
        "token": token,
        "user_id": oid,
        "remember": bool(remember),
        # 把有效期宽度存进会话：touch_session 续期时据此计算"剩余是否不足一半"，
        # 不必再去比对全局常量（"记住我"和普通会话宽度不同，不存就没法区分）
        "max_age": int(max_age),
        "created_at": now,
        "last_seen_at": now,
        "expires_at": now + timedelta(seconds=max_age),
    }
    try:
        db = _get_db()
        db[SESSIONS_COLLECTION].insert_one(doc)
    except AuthError:
        raise
    except PyMongoError as e:
        raise AuthStorageError(f"创建会话失败（登录存储异常）：{e}") from e

    logger.info(f"[auth] 会话已创建：user_id={oid} remember={bool(remember)} max_age={max_age}s")
    return token, (_public_session(doc) or {})


def _resolve_session(token: Optional[str]) -> Tuple[Optional[dict], Optional[dict]]:
    """内部校验：返回 (session_doc, public_user_doc)；无效/过期都返回 (None, None)。

    同时读 user 文档而不是只回 session：会话有效但用户已被删除时，
    会话必须视为无效（否则删号之后旧 cookie 还能用满 30 天）。
    """
    if not token or not isinstance(token, str):
        return None, None
    try:
        db = _get_db()
        sessions = db[SESSIONS_COLLECTION]
        doc = sessions.find_one({"token": token})
    except AuthError:
        raise
    except PyMongoError as e:
        raise AuthStorageError(f"会话校验失败（登录存储异常）：{e}") from e

    if doc is None:
        return None, None

    now = _utcnow()
    expires_at = _as_utc(doc.get("expires_at"))
    if expires_at is None or expires_at <= now:
        # 代码层再判一次过期：TTL 后台线程约每 60 秒才清一轮，这段窗口内记录还在库里
        # 顺手删掉它（失败无所谓，反正已经判定无效），避免过期 token 被反复命中
        try:
            sessions.delete_one({"_id": doc["_id"]})
        except PyMongoError:
            pass
        logger.debug("[auth] 会话已过期，判定无效")
        return None, None

    user = get_user_by_id(doc.get("user_id"))
    if user is None:
        try:
            sessions.delete_one({"_id": doc["_id"]})
        except PyMongoError:
            pass
        logger.info("[auth] 会话对应的用户已不存在，会话同时作废")
        return None, None
    return doc, user


def get_session(token: str | None) -> dict | None:
    """按 token 取当前会话，有效则返回 {"user": {...}, "session": {...}}，否则 None。

    过期 / 伪造 / 空 / 用户已删除 → 一律 None（不抛，除非 Mongo 故障抛 AuthStorageError）。
    注意：本函数**不写库**（不做滑动续期），续期只在 touch_session 里发生。
    """
    doc, user = _resolve_session(token)
    if doc is None or user is None:
        return None
    return {"user": user, "session": _public_session(doc) or {}}


# touch_session 的写库节流表：token -> 上次评估时刻（monotonic）
_touch_lock = threading.Lock()
_touch_cache: dict[str, float] = {}


def _touch_from_doc(session_doc: Optional[dict]) -> None:
    """基于已读出的会话文档做滑动续期；**只在剩余时间不足一半时写一次库**。

    为什么这么设计（两个"为什么"）：
    1. 为什么不每请求写库：活跃前端每秒轮询 /status，一个会话就能打出几千次 UPDATE，
       鉴权层自己变成 Mongo 写放大源；按"剩余寿命 < 50% 才续"，一次续期能再撑半个周期，
       普通 12h 会话大约每 6 小时才写一条更新。
    2. 为什么还要再加 60 秒内存节流：即使不写库，评估本身也需要一次会话读；
       同一 token 60 秒内重复评估直接跳过，把高频轮询的成本压到内存。
    取舍：滑动续期意味着"持续活跃的会话永不过期"（被盗 token 只要一直用就一直有效）。
    这是"体验 vs 失窃窗口"的标准折中，用 REMEMBER_MAX_AGE=30 天给记住我的会话封顶；
    要更严格可以让 create_session 的 remember=False 分支改成不续期。
    本函数**永不抛异常**（续期是锦上添花，不该把已登录用户踢下线）。
    """
    try:
        if not session_doc:
            return
        token = session_doc.get("token")
        if not isinstance(token, str) or not token:
            return
        now_mono = time.monotonic()
        with _touch_lock:
            if now_mono - _touch_cache.get(token, 0.0) < _TOUCH_THROTTLE_SECONDS:
                return
            if len(_touch_cache) > _TOUCH_CACHE_MAX_KEYS:
                # 粗粒度回收：最坏情况是多写几条 UPDATE，比无界增长好
                _touch_cache.clear()
            _touch_cache[token] = now_mono

        now = _utcnow()
        expires_at = _as_utc(session_doc.get("expires_at"))
        if expires_at is None:
            return
        max_age = int(session_doc.get("max_age") or SESSION_MAX_AGE)
        remaining = (expires_at - now).total_seconds()
        if remaining <= 0 or remaining >= max_age / 2.0:
            return  # 还早，不写库
        db = _get_db()
        db[SESSIONS_COLLECTION].update_one(
            {"token": token},
            {"$set": {"expires_at": now + timedelta(seconds=max_age), "last_seen_at": now}},
        )
        logger.debug(f"[auth] 会话滑动续期：剩余 {int(remaining)}s -> {max_age}s")
    except Exception as e:  # 刻意兜住一切：续期失败绝不能变成用户可见错误
        logger.debug(f"[auth] 会话续期跳过（不影响本次请求）：{e}")


def touch_session(token: str) -> None:
    """滑动续期的公开入口：需要多一次会话读（get_session 之后请依赖 require_user 的续期）。

    契约：**不抛异常**；无效/过期 token 静默无操作。
    """
    if not token or not isinstance(token, str):
        return
    try:
        db = _get_db()
        doc = db[SESSIONS_COLLECTION].find_one(
            {"token": token},
            projection={"token": 1, "expires_at": 1, "max_age": 1},
        )
    except Exception as e:
        logger.debug(f"[auth] touch_session 读取跳过：{e}")
        return
    _touch_from_doc(doc)


def delete_session(token: str | None) -> bool:
    """注销会话（退出登录）：返回是否真的删掉了一条记录。

    空 token / 已不存在的 token 返回 False，**不抛**。
    Mongo 故障抛 AuthStorageError：退出登录时"没删掉旧会话"是有安全含义的事
    （凭据仍然有效），调用方需要知道，不能当成"已退出"。
    """
    if not token or not isinstance(token, str):
        return False
    try:
        db = _get_db()
        result = db[SESSIONS_COLLECTION].delete_one({"token": token})
    except AuthError:
        raise
    except PyMongoError as e:
        raise AuthStorageError(f"注销失败（登录存储异常）：{e}") from e
    deleted = bool(result and result.deleted_count)
    with _touch_lock:
        _touch_cache.pop(token, None)
    if deleted:
        logger.info("[auth] 会话已注销")
    return deleted


# ====================
# Cookie 读写（把 flag 收敛到一处，避免每个接口自己拼一遍拼错）
# ====================


def set_session_cookie(response: Response, token: str, remember: bool = False) -> None:
    """把会话 token 写进 cookie：HttpOnly + SameSite=Lax + Path=/ 。

    为什么这三个 flag 缺一不可：
    * HttpOnly：document.cookie 读不到，一次 XSS 也就偷不走会话凭据（最关键）。
    * SameSite=Lax：跨站 POST/iframe 不带 cookie，挡住经典 CSRF；
      刻意不用 Strict —— Strict 会让用户从微信/邮件链接点进来的首次导航不带 cookie，
      表现为"莫名被踢回登录页"。Lax 仍保留顶级导航（GET）带 cookie 的行为。
    * Path=/：cookie 要同时给 8001 的页面与 /api 接口用，限定子路径会漏。
    * max_age 显式设置：浏览器重启后仍能保持登录（否则变成"关掉浏览器就没会话"）。
    * secure 由 AUTH_COOKIE_SECURE 控制：本机明文 HTTP 必须留 0，否则浏览器直接丢 cookie。
    """
    if not isinstance(response, Response) or not token:
        return
    max_age = REMEMBER_MAX_AGE if remember else SESSION_MAX_AGE
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=max_age,
        path="/",
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(),
    )


def clear_session_cookie(response: Response) -> None:
    """删除 cookie（退出登录时与 delete_session 配套调用）。

    delete_cookie 只让浏览器忘掉；服务端记录必须同时删（delete_session），
    否则拿到旧 cookie 的请求还能凭库里的记录继续通过校验 —— 两步都要做。
    """
    if not isinstance(response, Response):
        return
    response.delete_cookie(key=SESSION_COOKIE, path="/", samesite="lax")


def read_session_token(request: Request) -> str | None:
    """从请求里取会话 token；没有任何 cookie 也**不抛**（返回 None）。

    取值顺序：cookie 优先，其次 `Authorization: Bearer <token>`。
    为什么额外支持 Bearer：curl / 自动化脚本调用业务接口时不必先跑一遍 cookie jar，
    自检与运维排障都更省事；前端不需要改任何东西（它只依赖浏览器自动带 cookie）。
    """
    raw: Any = None
    try:
        raw = request.cookies.get(SESSION_COOKIE)
    except Exception:
        # 传进来的对象没有 .cookies（极少数测试脚本的假 request）：不抛，转去试 header
        raw = None
    if isinstance(raw, str) and raw.strip():
        return _finalize_token(raw.strip())
    return _bearer_token(request)


def _bearer_token(request: Request) -> Optional[str]:
    """从 Authorization: Bearer 头取 token；没有或格式不对返回 None（不抛）。"""
    header: Any = None
    try:
        header = request.headers.get("authorization", "")
    except Exception:
        return None
    if not isinstance(header, str) or not header.lower().startswith("bearer "):
        return None
    return _finalize_token(header[7:].strip())


def _finalize_token(token: Optional[str]) -> Optional[str]:
    """长度守卫：空串或异常超长（>128）的 token 视为不存在，省掉一次无意义的索引查询。"""
    if not token:
        return None
    if len(token) > _TOKEN_MAX_LEN:
        return None
    return token


# ====================
# FastAPI 依赖：页面 / 业务接口的门禁
# ====================


def is_html_request(request: Request) -> bool:
    """判定"这是浏览器地址栏/标签页导航请求"，用于决定 302 还是 401。

    规则：Accept 头含 text/html，且路径不以 /api/ 开头。

    为什么要区分：未登录时，
      - 导航请求应该 302 到登录页（用户看到的是登录界面）；
      - fetch/XHR 必须回 401（前端能读到状态码并弹提示）。
    若给 XHR 也回 302，浏览器会静默跟随重定向，前端拿到的是登录页的 **HTML 200**，
    于是 `res.json()` 抛异常、界面卡在"加载中" —— 最难查的一类假故障。
    为什么排除 /api/：前端有时用 `fetch(..., headers:{Accept:'*/*'})`，但也有些
    页面脚本直接发 Accept: text/html；显式的 /api/ 前缀是更硬的信号。
    （EventSource 的 Accept 是 text/event-stream → 归 API，返回 401，符合预期。）
    """
    try:
        accept = request.headers.get("accept", "") or ""
        path = request.url.path or ""
    except Exception:
        return False
    if path.startswith("/api/"):
        return False
    return "text/html" in accept.lower()


def require_user(request: Request) -> dict:
    """FastAPI 依赖：必须有有效会话，否则拦下来。

    分支（严格按需求）：
    * 无有效会话 + HTML 导航请求 → HTTPException(302, headers={"Location": LOGIN_PATH})
    * 无有效会话 + API 请求      → HTTPException(401, detail="未登录或会话已过期")
    * Mongo 故障                 → HTTPException(503)（不把存储故障说成"未登录"）
    通过校验时顺带做滑动续期（自带 60 秒节流，不额外抛）。

    用法：`async def page(user: dict = Depends(require_user))`
    """
    token = read_session_token(request)
    try:
        session_doc, user = _resolve_session(token)
    except AuthStorageError as e:
        raise HTTPException(
            status_code=503,
            detail=f"登录服务暂时不可用（鉴权存储不可达），请稍后重试：{e}",
        ) from e

    if session_doc is None or user is None:
        if is_html_request(request):
            # 用 302 而不是 RedirectResponse：require_user 是依赖函数，
            # 只能通过 HTTPException 的 headers 让 FastAPI 拼出重定向，语义等价。
            # 不加 ?next= ：需求给定的 Location 就是裸 LOGIN_PATH，
            # 由登录页自己决定成功后回哪儿（也避免把任意 URL 反射进跳转造成开放重定向）。
            raise HTTPException(status_code=302, headers={"Location": LOGIN_PATH})
        raise HTTPException(status_code=401, detail="未登录或会话已过期")

    _touch_from_doc(session_doc)
    return user


def optional_user(request: Request) -> dict | None:
    """FastAPI 依赖：登录了就给 user，没登录/过期/Mongo 故障都给 None（永不拦请求）。

    为什么 Mongo 故障在这里也返回 None：本函数的契约是"不拦"，
    用它做门禁的接口本来就不依赖身份；把故障升级成 503 反而会让匿名可用功能挂掉。
    真正的门禁请挂 require_user。
    """
    token = read_session_token(request)
    if not token:
        return None
    try:
        session_doc, user = _resolve_session(token)
    except AuthError as e:
        logger.warning(f"[auth] optional_user 解析会话失败（按未登录处理）：{e}")
        return None
    if session_doc is None or user is None:
        return None
    _touch_from_doc(session_doc)
    return user


# ====================
# 健康检查 / 排障
# ====================


def auth_status() -> dict:
    """给 /health 与排障用的鉴权子系统状态：{"mongo": bool, "users": int, "detail": str}。

    刻意只返回**计数**：绝不返回任何用户文档，也就不可能泄露 password_hash
    （日志与响应里都没有散列，见本文件的日志纪律）。
    本函数不抛异常：探测函数在 Mongo 挂掉时必须给出"挂掉"这个结论，而不是 500。
    """
    url = _mongo_url()
    db_name = _mongo_db_name()
    try:
        db = _get_db()
        db.command("ping")
        users = int(db[USERS_COLLECTION].count_documents({}))
        sessions = int(db[SESSIONS_COLLECTION].count_documents({}))
        detail = (
            f"ok: db={db_name} users={users} active_sessions={sessions} "
            f"indexes={'ready' if _indexes_ready else 'pending'} "
            f"cookie_secure={'on' if _cookie_secure() else 'off'}"
        )
        return {"mongo": True, "users": users, "detail": detail}
    except Exception as e:
        # 只带异常类名和一句话，不打堆栈（/health 的响应可能被前端 console 打印）
        logger.warning(f"[auth] auth_status 探测失败：{type(e).__name__}: {e}")
        return {
            "mongo": False,
            "users": 0,
            "detail": (
                f"登录存储不可用（{type(e).__name__}）：{e}；"
                f"请检查 Mongo 服务与 MONGO_URL={url} / MONGO_DB_NAME={db_name}"
            ),
        }


if __name__ == "__main__":
    # 手工自检入口：只打印状态与一次散列耗时，不写库、不打印任何散列值。
    # 完整自检（建用户/会话/限流/降级/302+401 分支）在临时脚本里跑，见交付报告。
    started = time.perf_counter()
    sample_hash = hash_password("example-password-123")
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"iterations={_PBKDF2_ITERATIONS} hash_password={elapsed_ms:.1f}ms")
    print(f"verify={verify_password('example-password-123', sample_hash)}")
    print(f"auth_status={auth_status()}")
