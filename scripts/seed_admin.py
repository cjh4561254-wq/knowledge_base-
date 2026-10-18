# scripts/seed_admin.py
"""
首个管理员账号 seed 脚本（鉴权上线后的"开门钥匙"）
=================================================

为什么需要这个脚本
------------------
`utils/auth_utils.py` 的 require_user 是硬门禁：users 集合一条记录都没有的时候，
chat / import 页面和全部业务接口都会把请求挡在门外，而登录页又没有注册入口
（管理员账号不该由公开注册接口产生）。所以在启用鉴权之前，必须有一个**只有运维能跑**
的入口，把第一个用户写进 users 集合。

设计约束（与 auth_utils 的契约对齐）
------------------------------------
* **绝不自己实现哈希**：PBKDF2 参数、盐、格式、恒定时间比较全在 auth_utils 里。
  本脚本只用 create_user / authenticate / hash_password / close / auth_status。
* `create_user` 返回的 `_id` 是**字符串**（不是 ObjectId）、重名抛 `UsernameTakenError`、
  Mongo 故障抛 `AuthStorageError` —— 这三件事在下面的分支里都被显式处理。
* 零新依赖：标准库（argparse / secrets / getpass / subprocess）+ 项目已有的 pymongo/bson。
* 只 import `utils.auth_utils`，**不 import web/***：否则会把 FastAPI 整条依赖链
  （模型、Milvus、MinIO）拖起来，一个建账号的脚本要跑几分钟。

随机口令的去向（安全要求）
--------------------------
控制台打印一次 + 写入项目根的 ADMIN_PASSWORD.txt（权限尽力收紧，且已被 .gitignore
忽略）。文件里除用户名与这一条临时口令之外**没有任何其它凭据**（.env 的 key 一律不抄），
文件头写着"仅本机、请立刻改掉并删除本文件"。
用 `--password` 自行指定的口令**不落盘** —— 不让明文口令永久留在磁盘上。
"""

from __future__ import annotations

import argparse
import getpass
import os
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path 兜底：脚本可能被从任意 cwd 运行。
# `python scripts/seed_admin.py` 时 sys.path[0] 是 scripts/ 而不是项目根，
# 此时 import utils.auth_utils 会直接 ModuleNotFoundError —— 所以把"本文件的
# 上一级目录"（= 项目根）显式插到 sys.path 最前面，让脚本能否运行与 cwd 无关。
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_root_str = str(PROJECT_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

try:
    from utils import auth_utils as au
except ImportError as exc:  # pragma: no cover - 只在项目根被搬动时出现
    sys.stderr.write(
        f"导入 utils.auth_utils 失败：{exc}\n"
        f"请确认本脚本位于项目内的 scripts/ 目录下（当前推断的项目根：{PROJECT_ROOT}）。\n"
    )
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 默认用户名（需求指定）
DEFAULT_USERNAME = "admin"

#: 默认展示名（create_user 的 display_name，仅用于人工识别，不参与鉴权判定）
DEFAULT_DISPLAY_NAME = "管理员"

#: 随机口令默认长度。刻意取 20（满足"16 位以上"，同时远离 auth_utils 的 128 上限）。
PASSWORD_LENGTH = 20

#: 随机口令最小长度（需求硬线）
PASSWORD_LENGTH_MIN = 16

#: 随机口令最大长度：与 auth_utils 的口令规则（8–128）对齐。
#: 刻意写成本地常量而不是去读 au._PASSWORD_MAX_LEN —— 那是私有名字，重构掉就会炸。
_PASSWORD_MAX_ACCEPTABLE = 128

#: 刻意排除的易混字符：I l 1 O 0，外加小写 o（多数等宽字体里它和 0 同形）
EXCLUDED_CHARS = set("Il1O0o")

_ALPHABET_UPPER = "".join(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in EXCLUDED_CHARS)
_ALPHABET_LOWER = "".join(c for c in "abcdefghijklmnopqrstuvwxyz" if c not in EXCLUDED_CHARS)
_ALPHABET_DIGIT = "".join(c for c in "0123456789" if c not in EXCLUDED_CHARS)
#: 符号集刻意避开引号/反斜杠/空白与 shell 特殊字符（& ; | < > ( ) # $ * ! ^ % 等）：
#: 管理员会把这个串粘进命令行或表单，特殊字符会让粘贴结果被 shell 改写甚至执行。
_ALPHABET_SYMBOL = "-_.:@+=?"
_ALPHABET_ALL = _ALPHABET_UPPER + _ALPHABET_LOWER + _ALPHABET_DIGIT + _ALPHABET_SYMBOL
_ALPHABET_ALNUM = _ALPHABET_UPPER + _ALPHABET_LOWER + _ALPHABET_DIGIT

#: 凭据文件固定在**项目根**（不是 cwd！），与 .gitignore 里新增忽略的那一条同名
PASSWORD_FILE = PROJECT_ROOT / "ADMIN_PASSWORD.txt"

#: 登录页地址（auth_utils.LOGIN_PATH 是契约常量，不在这儿硬编码路径名）
LOGIN_URL = f"http://127.0.0.1:8001{au.LOGIN_PATH}"

#: Mongo 连不上时给的一句话排查提示（用的是本项目 compose 里的真实服务名）
MONGO_HINT = (
    "请先启动登录存储：docker compose -f deploy/docker-compose.yml up -d kb-mongo"
    "（本机默认 mongodb://localhost:27017、库名 kb001；改过就核对 .env 的 "
    "MONGO_URL / MONGO_DB_NAME）"
)

# ---------------------------------------------------------------------------
# 输出小工具
# ---------------------------------------------------------------------------


def say(text: str = "") -> None:
    """打印一行中文输出。

    为什么要包一层：Windows 控制台默认代码页是 GBK，输出里出现中文之外的符号（箭头、
    生僻标点）时 print 会抛 UnicodeEncodeError，把"建账号成功"演成异常退出。
    这里编码失败就退化成 ASCII，绝不因为编码把结果弄丢。
    """
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def section(title: str) -> None:
    say("")
    say(f"===== {title} =====")


def is_interactive() -> bool:
    """是否具备可交互的终端。

    为什么重要：非交互（管道 / CI / 本脚本的自检）下绝不能再调 getpass/input 等输入，
    那会让进程永远挂在那里。需求要的是"给清晰报错而不是卡住"。

    为什么 stdin 和 stdout **都要**看：只看 stdin 不够可靠 —— 实测在 Git Bash 里把
    stdin 重定向成 /dev/null（Windows 侧映射为 NUL 设备）时，isatty() 依然返回 True，
    于是脚本会去调 input() 并立刻抛 EOFError（自检时就踩到了这条）。要求 stdout 也是
    终端，才能把"真的有人在打字"和"有人在管道里喂数据"区分开；判断不清时一律按
    非交互处理 —— 对 --reset / --delete 这类破坏性动作，fail-closed 才是对的方向。
    """
    try:
        if sys.stdin is None or sys.stdin.closed:
            return False
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (ValueError, OSError):
        return False


def confirm_or_pass(prompt: str, assume_yes: bool) -> bool:
    """二次确认。--yes 预授权直接放行；交互终端要求手打 yes。

    读不到输入（EOF / 无控制台）时按"未确认"返回 False：确认类动作只在明确 yes 时
    才放行，绝不允许"读输入失败 = 默认同意"这种危险的默认。
    """
    if assume_yes:
        return True
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt, OSError, RuntimeError) as exc:
        say(f"（提示）读不到确认输入（{type(exc).__name__}），已按「未确认」处理。")
        return False
    return answer == "yes"


# ---------------------------------------------------------------------------
# 底层数据访问：--list / --delete / 重置密码需要的批量读写，auth_utils 没有公开函数
# ---------------------------------------------------------------------------

_db_handle = None
_own_client = None  # 仅当走"自建连接"兜底分支时才非 None


def database():
    """拿 users/sessions 所在 db 句柄（延迟建连；失败由 auth_utils 抛 AuthStorageError）。

    正常路径复用 auth_utils 的单例连接：同一套 5 秒超时与建索引逻辑，不额外开连接池。
    兜底分支只在 `_get_db` 这个内部名字因重构消失时生效：按 auth_utils 已经
    `load_dotenv(override=True)` 过的环境变量自建一条连接，至少脚本不至于当场废掉。
    """
    global _db_handle, _own_client

    if _db_handle is not None:
        return _db_handle

    getter = getattr(au, "_get_db", None)
    if callable(getter):
        _db_handle = getter()
        return _db_handle

    from pymongo import MongoClient

    url = (os.getenv("MONGO_URL") or "mongodb://localhost:27017").strip()
    db_name = (os.getenv("MONGO_DB_NAME") or "kb001").strip() or "kb001"
    timeout_ms = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000"))
    _own_client = MongoClient(
        url,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
        socketTimeoutMS=timeout_ms,
    )
    _db_handle = _own_client[db_name]
    return _db_handle


def shutdown() -> None:
    """释放连接：先关自建兜底连接，再调 auth_utils.close()（它本身幂等）。"""
    global _db_handle, _own_client
    if _own_client is not None:
        try:
            _own_client.close()
        except Exception:  # 兜底连接关闭失败无关紧要
            pass
        _own_client = None
    _db_handle = None
    try:
        au.close()
    except Exception as exc:  # close 设计上不抛，这里只是防止退出路径被卡住
        say(f"（提示）释放 Mongo 连接时出现异常，可忽略：{exc}")


def to_object_id(raw) -> object:
    """字符串 _id → ObjectId。

    auth_utils 对外把 _id 转成字符串（契约），所以按 _id 定位文档时必须自己转回去；
    非法值一律抛 AuthError（可读文案），绝不用一个可疑的 id 去 delete_many。
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        return ObjectId(str(raw))
    except (InvalidId, TypeError, ValueError) as exc:
        raise au.AuthError(f"用户 _id 不是合法 ObjectId（值={raw!r}），为避免误操作已中止") from exc


def count_users() -> int:
    """users 集合计数（--verify 之外，前后对比也用这个数）。"""
    return int(database()[au.USERS_COLLECTION].count_documents({}))


def list_usernames() -> list[str]:
    """列出所有用户名。

    刻意用 projection 只取 username 字段：即便以后文档里多了别的敏感字段，
    这个函数在**代码层面**就拿不到 password_hash，而不是靠"记得别打印"。
    """
    docs = database()[au.USERS_COLLECTION].find({}, projection={"username": 1})
    return sorted({str(d.get("username")) for d in docs if d.get("username")})


# ---------------------------------------------------------------------------
# 随机强口令
# ---------------------------------------------------------------------------


def generate_password(length: int = PASSWORD_LENGTH) -> str:
    """生成 length 位强口令：四类字符各至少 1 个、排除易混字符、首位是字母或数字。

    熵：字母集约 70 个字符、长度 20 → log2(70^20) ≈ 118 bit，人工/离线爆破都不可行。
    随机源用 secrets（CSPRNG）；打乱也用 SystemRandom —— `random` 是伪随机、可被预测，
    绝不能用来生成口令。
    """
    length = max(PASSWORD_LENGTH_MIN, int(length))
    rng = secrets.SystemRandom()
    # 先保证四类各出现一次，否则短口令可能整类缺失（策略校验会打回）
    chars = [
        rng.choice(_ALPHABET_UPPER),
        rng.choice(_ALPHABET_LOWER),
        rng.choice(_ALPHABET_DIGIT),
        rng.choice(_ALPHABET_SYMBOL),
    ]
    chars += [rng.choice(_ALPHABET_ALL) for _ in range(length - len(chars))]
    rng.shuffle(chars)

    # 首位刻意取字母/数字：`--password -abc` 会被 argparse 当成选项，
    # 个别表单也会把开头的符号当占位符处理。
    if chars[0] not in _ALPHABET_ALNUM:
        for i in range(1, len(chars)):
            if chars[i] in _ALPHABET_ALNUM:
                chars[0], chars[i] = chars[i], chars[0]
                break
    password = "".join(chars)

    # 自检：生成的串必须过 auth_utils 自己的口令长度规则（8–128），且不含易混字符
    if len(password) < PASSWORD_LENGTH_MIN or len(password) > _PASSWORD_MAX_ACCEPTABLE:
        raise RuntimeError(f"生成的口令长度不合规（{len(password)}），已中止（未写库）")
    if EXCLUDED_CHARS & set(password):
        raise RuntimeError("生成的口令含易混字符，已中止（未写库）")
    for alphabet in (_ALPHABET_UPPER, _ALPHABET_LOWER, _ALPHABET_DIGIT, _ALPHABET_SYMBOL):
        if not any(c in alphabet for c in password):
            raise RuntimeError("生成的口令缺少必需字符类别，已中止（未写库）")
    return password


# ---------------------------------------------------------------------------
# 凭据文件
# ---------------------------------------------------------------------------


def tighten_permissions(path: Path) -> tuple[str, str]:
    """尽力把凭据文件权限收紧为"仅当前用户"。永不抛异常。

    返回 (状态, 说明)，状态取 ok / failed。

    为什么必须分平台：POSIX 上 chmod 0o600 是真有效的；Windows 的 os.chmod 只映射到
    "只读位"，**没有 600 这个概念**，实际权限由 NTFS ACL 决定。所以 Windows 走 icacls：
    先断开继承（/inheritance:r），再只授权当前用户读写（/grant:r）。
    失败不影响建账号这件事，只要把补救命令原样告诉使用者。
    """
    if os.name == "nt":
        user = (os.environ.get("USERNAME") or "").strip()
        if not user:
            return "failed", "环境变量 USERNAME 为空，无法自动执行 icacls，请手工收紧权限"
        cmd = ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:M"]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
            stdout = (proc.stdout or b"").decode("mbcs", "replace")
            stderr = (proc.stderr or b"").decode("mbcs", "replace")
        except (OSError, subprocess.SubprocessError) as exc:
            return "failed", f"icacls 无法执行（{type(exc).__name__}: {exc}）"
        if proc.returncode == 0:
            return "ok", f"icacls 已断开权限继承，只授权 {user} 修改（M）"
        detail = (stderr or stdout).strip().splitlines()
        return "failed", "icacls 失败：" + (detail[-1] if detail else f"返回码 {proc.returncode}")

    try:
        os.chmod(path, 0o600)
        return "ok", "chmod 600 已生效"
    except OSError as exc:
        return "failed", f"chmod 失败（{exc}）"


def write_password_file(username: str, password: str, kind: str) -> bool:
    """把临时口令写入 ADMIN_PASSWORD.txt（只有脚本生成的口令才走这里）。

    文件内容刻意只包含"用户名 + 这一条临时口令"：MONGO_URL、模型 API key 之类的
    .env 凭据一个都不抄 —— 免得一个 txt 变成全仓库的凭据汇总点。
    """
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    content = f"""=======================================================================
 智能知识库 · 管理员临时凭据（由 scripts/{Path(__file__).name} 生成）
=======================================================================
!!! 安全提醒：请现在就读完下面三行 !!!
  1. 这是**仅本机**的一次性临时口令。请立刻登录 {LOGIN_URL}
     改掉密码，改完之后**删除本文件**（不要留着，也不要转存到聊天记录/截图里）。
  2. 本文件已被 .gitignore 忽略（ADMIN_PASSWORD.txt）。不要 git add -f 强行提交。
  3. 除用户名与下面这一条临时口令之外，本文件不含任何其它凭据
     （.env 里的 MONGO_URL / API key 等一律没有抄）。

 权限说明：Windows 上 chmod 600 不起作用（它只影响只读位），实际权限由 NTFS ACL 决定。
 本脚本会自动执行一次收紧；若下面记录的结果是失败，请在项目根手工执行（效果等价）：
     icacls ADMIN_PASSWORD.txt /inheritance:r /grant:r "%USERNAME%:M"

-----------------------------------------------------------------------
 生成时间(本地) : {now}
 事件类型       : {kind}
 用户名         : {username}
 临时密码       : {password}
-----------------------------------------------------------------------
 用完即删。删除命令：CMD  del ADMIN_PASSWORD.txt   /  PowerShell  Remove-Item ADMIN_PASSWORD.txt
=======================================================================
"""
    try:
        PASSWORD_FILE.write_text(content, encoding="utf-8")
    except OSError as exc:
        say(f"（异常）凭据文件写入失败：{type(exc).__name__}: {exc}")
        say(f"   预期路径：{PASSWORD_FILE}")
        say(f"   本次口令只在控制台出现这一次，请务必另存：{password}")
        return False

    status, note = tighten_permissions(PASSWORD_FILE)
    say(f"凭据文件：{PASSWORD_FILE}")
    if status == "ok":
        say(f"  权限收紧结果：{note}")
    else:
        say(f"  权限收紧结果：未成功 —— {note}")
        say('  请手工执行：icacls ADMIN_PASSWORD.txt /inheritance:r /grant:r "%USERNAME%:M"')
    say("  必须提醒：请**马上登录改掉这个密码，然后删除本文件**。")
    return True


# ---------------------------------------------------------------------------
# 交互取密码
# ---------------------------------------------------------------------------


def prompt_password_interactively(label: str) -> str:
    """getpass 取口令（不回显），并要求二次输入一致。

    正常只在交互终端被调用（上游用 is_interactive() 拦过），但 getpass 在没有真实
    控制台时仍可能抛 EOFError / OSError（Windows 无 console 窗口时是 "GetPassword
    failed"），这里统一转成"清晰报错 + 退出"，绝不让脚本卡住或甩裸栈。
    """
    try:
        first = getpass.getpass(f"{label}（输入不回显）: ")
        second = getpass.getpass("请再次输入以确认: ")
    except (EOFError, KeyboardInterrupt, OSError, RuntimeError) as exc:
        say(f"[错误] 无法读取口令（{type(exc).__name__}: {exc}）—— 当前环境没有可用的终端。")
        say("       请改用：--password=\"你的口令\"（直接给值），"
            "或完全不传 --password 让脚本随机生成强口令。")
        raise SystemExit(2)
    if first != second:
        say("[错误] 两次输入不一致，已中止（未对数据库做任何改动）。")
        raise SystemExit(2)
    return first


# ---------------------------------------------------------------------------
# 动作实现
# ---------------------------------------------------------------------------


def action_verify(_: argparse.Namespace) -> int:
    """--verify：不写库，只打印 auth_status() 与 users 数量。"""
    section("鉴权存储自检（只读，不写库）")
    status = au.auth_status()
    say(f"Mongo 可用      : {'是' if status.get('mongo') else '否'}")
    say(f"users 用户数    : {status.get('users')}")
    say(f"detail          : {status.get('detail')}")
    say(f"库名 / 集合      : {os.getenv('MONGO_DB_NAME') or 'kb001'} / "
        f"{au.USERS_COLLECTION}, {au.SESSIONS_COLLECTION}")
    say(f"会话 cookie     : {au.SESSION_COOKIE}（{au.SESSION_MAX_AGE}s，"
        f"记住我 {au.REMEMBER_MAX_AGE}s）")
    if not status.get("mongo"):
        say("")
        say("[错误] 登录存储不可用 —— 启用鉴权后所有人都会被挡在门外。")
        say(f"       {MONGO_HINT}")
        return 1
    if not status.get("users"):
        say("")
        say("（提示）当前 0 个用户：require_user 会拦住全部页面与接口。请立即执行 "
            "python scripts/seed_admin.py 建首个管理员账号。")
    return 0


def action_list(_: argparse.Namespace) -> int:
    """--list：只列用户名，绝不打印口令或其散列。"""
    section("用户清单（仅用户名）")
    names = list_usernames()
    say(f"共 {len(names)} 个用户：")
    for i, name in enumerate(names, 1):
        say(f"  {i}. {name}")
    if not names:
        say("  （空 —— 一个用户都没有，任何页面都进不去；请先跑 seed_admin.py）")
    say("")
    say("说明：本命令只投影 username 字段，不读取也不输出密码或密码散列。")
    return 0


def action_delete(args: argparse.Namespace) -> int:
    """--delete NAME：删除用户及其全部会话（不可逆，必须确认）。"""
    username = au.normalize_username(args.delete)
    if not username:
        say("[错误] --delete 需要一个非空用户名。")
        return 2

    section(f"删除用户：{username}")
    user = au.get_user_by_username(username)
    if user is None:
        say(f"用户「{username}」不存在，未做任何改动。（用 --list 看现有用户名）")
        return 1

    if not args.yes and not is_interactive():
        say("非交互环境：删除用户必须显式加 --yes。这是不可逆操作，刻意不做默认放行。")
        say(f"  python scripts/{Path(__file__).name} --delete {username} --yes")
        return 2
    if not confirm_or_pass(
        f"确认删除用户「{username}」及其全部会话？输入 yes 继续： ", args.yes
    ):
        say("已取消，未删除任何数据。")
        return 1

    db = database()
    try:
        oid = to_object_id(user.get("_id"))
    except au.AuthError as exc:
        say(f"[错误] {exc}")
        return 1

    sessions_deleted = int(db[au.SESSIONS_COLLECTION].delete_many({"user_id": oid}).deleted_count)
    users_deleted = int(db[au.USERS_COLLECTION].delete_one({"_id": oid}).deleted_count)
    say(f"users 删除 {users_deleted} 条；sessions 删除 {sessions_deleted} 条（旧会话同时作废）。")
    say(f"剩余用户数：{count_users()}")
    if users_deleted:
        say(f"用户「{username}」已删除。")
        return 0
    say(f"[错误] 没有删除到用户「{username}」，请再核对用户名。")
    return 1


def action_seed(args: argparse.Namespace) -> int:
    """默认动作：创建/确保管理员账号；给了 --reset 才重置口令。

    幂等口径（需求明确）：用户已存在且没给 --reset → 打印"已存在，未改动"，
    **不生成新口令、不覆盖凭据文件**。
    """
    username = au.normalize_username(args.username)
    if not username:
        say("[错误] 用户名不能为空。")
        return 2
    say(f"目标用户名：{username}"
        "（auth_utils 会做归一化与校验：去空白、转小写、3–32 位、仅中文/字母/数字/._-）")

    existing = au.get_user_by_username(username)

    # ---- 分支 1：已存在且未要求重置 → 完全不动 ----
    if existing is not None and not args.reset:
        section("结果：已存在，未改动")
        say(f"用户「{username}」已经存在（_id={existing.get('_id')}，"
            f"display_name={existing.get('display_name') or '-'}）。")
        say("未生成新密码，未修改数据库，也没有覆盖凭据文件。")
        say("确需重置密码请显式加 --reset（非交互环境还需 --yes）：")
        say(f"  python scripts/{Path(__file__).name} --username {username} --reset --yes")
        say(f"当前 users 总数：{count_users()}")
        return 0

    resetting = existing is not None and args.reset
    section("重置密码" if resetting else "创建用户")

    if resetting:
        if not args.yes and not is_interactive():
            say("非交互环境：重置已有账号的密码必须显式加 --yes，"
                "避免一条手滑的命令就把管理员口令换掉。")
            say(f"  python scripts/{Path(__file__).name} --username {username} --reset --yes")
            return 2
        if not confirm_or_pass(
            f"确认把用户「{username}」的密码重置为新密码？旧密码将立即失效。输入 yes 继续： ",
            args.yes,
        ):
            say("已取消，未改动该用户。")
            return 1
        say("（说明）重置只换口令；sessions 里已有的会话仍然有效，直到过期或注销。")

    # --password 缺省（None）→ 随机生成；显式传空串 → 交互输入
    generated = args.password is None
    if generated:
        password = generate_password(args.password_length)
        kind = "重置时脚本随机生成" if resetting else "脚本随机生成"
    elif args.password == "":
        if not is_interactive():
            say("[错误] --password 传了空值，含义是「交互式输入」，但当前 stdin 不是终端"
                "（非交互环境）。继续下去会一直卡在等输入，所以这里直接退出。")
            say("       两种解决办法：① 直接给值 --password=\"你的密码\"；"
                "② 完全不传 --password，让脚本随机生成强口令并写入 ADMIN_PASSWORD.txt。")
            return 2
        password = prompt_password_interactively(f"请为用户「{username}」设置密码")
        kind = "交互输入（不写入文件）"
    else:
        password = args.password
        kind = "命令行指定（不写入文件）"

    if resetting:
        return apply_reset(username, existing, password, generated, kind)
    return apply_create(username, password, args.display_name, generated, kind)


def apply_create(username: str, password: str, display_name: str, generated: bool, kind: str) -> int:
    """新建用户 → 立刻用 authenticate 自证 → 按需落盘临时口令。"""
    try:
        user = au.create_user(username, password, display_name)
    except au.UsernameTakenError as exc:
        # 撞上 users.username 唯一索引（并发/重复执行）：诚实反馈"已存在"，绝不覆盖文件
        section("结果：已存在，未改动")
        say(str(exc))
        say("本次没有生成新密码，也没有改动任何文件或数据库记录。")
        say("确需换口令请加 --reset --yes。")
        return 1
    except (au.InvalidUsernameError, au.InvalidPasswordError) as exc:
        say(f"[错误] 参数不合规：{exc}")
        return 2
    except au.AuthStorageError as exc:
        report_storage_error(exc)
        return 1

    say(f"已创建用户：username={user.get('username')} _id={user.get('_id')} "
        f"display_name={user.get('display_name') or '-'}")
    say("（_id 是字符串 —— 这是 auth_utils 的对外契约，文档不返回 ObjectId）")
    return finish(username, password, generated, kind, action_word="创建")


def apply_reset(username: str, existing: dict, password: str,
                generated: bool, kind: str) -> int:
    """重置已有用户的口令。

    为什么不自己算散列、也不走 create_user（名字已占用必然抛 UsernameTakenError）：
    哈希参数（算法、迭代次数、盐长、串格式）属于 auth_utils 的私有决策，脚本自己
    update 一份手写散列等于把口令策略分叉成两套，以后它把 1_000_000 提上去就漏改这里。
    做法：**调 au.hash_password 生成散列**（不碰任何参数），再按 _id 原子替换那一个字段。

    审计字段刻意写成 ISO **字符串**（password_updated_at）而不是用 Mongo 的
    $currentDate：后者存进来的是 BSON datetime，而 auth_utils._public_user() 只对
    created_at / last_login_at 做 ISO 化，多出来一个 datetime 字段就会原样出现在
    authenticate() 的返回里，并行改 web/* 的登录接口把它塞进 JSON 响应时要多踩一次坑。
    """
    oid_raw = existing.get("_id")
    if not oid_raw:
        say("[错误] 取不到该用户的 _id，放弃重置（数据库未改动）。")
        return 1
    try:
        oid = to_object_id(oid_raw)
        new_hash = au.hash_password(password)
        result = database()[au.USERS_COLLECTION].update_one(
            {"_id": oid},
            {
                "$set": {
                    "password_hash": new_hash,
                    "password_updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
    except au.AuthStorageError as exc:
        report_storage_error(exc)
        return 1
    except au.AuthError as exc:
        say(f"[错误] 重置失败：{exc}（数据库未改动）")
        return 1
    if not result.matched_count:
        say(f"[错误] 没找到 _id={oid_raw} 的用户，重置未发生。")
        return 1
    say(f"已重置用户「{username}」的口令（散列由 auth_utils.hash_password 生成，"
        f"匹配 {result.matched_count} 条、更新 {result.modified_count} 条）。")
    return finish(username, password, generated, kind, action_word="重置")


def finish(username: str, password: str, generated: bool, kind: str, action_word: str) -> int:
    """统一收尾：authenticate 自证 + 口令去向 + 登录地址。"""
    section("自证：用刚设置的口令登录一次")
    try:
        ok = au.authenticate(username, password) is not None
    except au.TooManyLoginAttemptsError as exc:
        say(f"[警告] 该用户名触发登录限流：{exc}")
        say("       口令本身已按上面的结果写入，等窗口过了再登录即可。")
        return 1
    except au.AuthError as exc:
        report_storage_error(exc)
        return 1
    say(f"authenticate({username!r}, <刚设置的口令>) = {ok}")
    if not ok:
        say("[错误] 登录校验没通过：库里的散列与这次口令对不上，请勿使用该账号并复查本次操作。")
        return 1

    section(f"结果：{action_word}成功")
    if generated:
        say("随机生成的临时密码（**只打印这一次**）：")
        say("")
        say(f"    用户名：{username}")
        say(f"    密  码：{password}")
        say("")
        write_password_file(username, password, kind)
    else:
        say(f"口令由你提供（{kind}），刻意不落盘：磁盘上没有明文副本。")
    say("")
    say(f"现在可以用该账号登录 {LOGIN_URL}")
    say(f"当前 users 总数：{count_users()}")
    return 0


def report_storage_error(exc: Exception) -> None:
    """把 Mongo 故障翻译成一句人话 + 下一步做什么，绝不甩裸栈。"""
    say("[错误] 登录存储（MongoDB）不可用，本次操作未完成。")
    say(f"       原因：{type(exc).__name__}: {exc}")
    say(f"       {MONGO_HINT}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    prog = f"python scripts/{Path(__file__).name}"
    parser = argparse.ArgumentParser(
        prog=prog,
        description="智能知识库：创建/校验/维护首个管理员账号。"
                    "口令散列一律由 utils/auth_utils.py 负责，本脚本不自己实现哈希。",
        epilog="示例：\n"
               f"  {prog}                                # 建 admin，随机强口令并写入 ADMIN_PASSWORD.txt\n"
               f"  {prog} --username ops --password=\"改成你的密码\"\n"
               f"  {prog} --username admin --reset --yes\n"
               f"  {prog} --list / --verify / --delete guest --yes\n"
               "说明：--password 不给值则随机生成；给空值则交互输入（需终端，非交互直接报错）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--username", metavar="NAME", default=DEFAULT_USERNAME,
                        help=f"用户名（默认 {DEFAULT_USERNAME}；规则 3–32 位中文/字母/数字/._-，自动小写归一化）")
    parser.add_argument("--password", metavar="PASS", nargs="?", default=None, const="",
                        help="指定口令。不传=随机生成强口令；传空值（--password=）=交互输入且不回显")
    parser.add_argument("--display-name", metavar="TEXT", default=DEFAULT_DISPLAY_NAME,
                        help=f"展示名（默认 {DEFAULT_DISPLAY_NAME}，不参与鉴权，最长 64 字符）")
    parser.add_argument("--reset", action="store_true",
                        help="用户已存在时重置其口令（需二次确认；非交互环境必须再加 --yes）")
    parser.add_argument("--yes", action="store_true",
                        help="为 --reset / --delete 预先确认（等价于交互终端里手打 yes）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有用户名（绝不输出口令或其散列）")
    parser.add_argument("--delete", metavar="NAME", default=None,
                        help="删除指定用户及其全部会话（需确认；非交互环境必须再加 --yes）")
    parser.add_argument("--verify", action="store_true",
                        help="只读：打印 auth_status() 与 users 数量，不写库")
    parser.add_argument("--password-length", metavar="N", type=int, default=PASSWORD_LENGTH,
                        help=f"随机口令长度（默认 {PASSWORD_LENGTH}，最小 {PASSWORD_LENGTH_MIN}）")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 动作互斥：--list / --verify / --delete 是三种互不相干的独立动作，同时给出就是自相矛盾
    chosen = [name for name, on in (
        ("--list", args.list),
        ("--verify", args.verify),
        ("--delete", args.delete is not None),
    ) if on]
    if len(chosen) > 1:
        parser.error(f"{chosen} 只能指定一个（一条命令只做一件事）。")
    if args.reset and (args.list or args.verify or args.delete is not None):
        parser.error("--reset 只属于默认（创建/确保用户）流程，不能与 --list/--verify/--delete 同用。")
    if args.yes and not (args.reset or args.delete is not None):
        say("（提示）--yes 只在配合 --reset / --delete 时才有意义，本次没有需要确认的操作。")

    say(f"项目根：{PROJECT_ROOT}")
    say(f"凭据文件位置（只有随机生成的口令才会写它）：{PASSWORD_FILE}")

    handler = (
        action_verify if args.verify
        else action_list if args.list
        else action_delete if args.delete is not None
        else action_seed
    )
    try:
        return int(handler(args))
    except au.AuthStorageError as exc:
        report_storage_error(exc)
        return 1
    except KeyboardInterrupt:
        say("")
        say("[中断] 已取消。若中断发生在写库之后，请用 --verify / --list 核对当前状态。")
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # 兜底：默认不甩裸栈；需要栈时设 KB_SEED_DEBUG=1
        say(f"[错误] 未预期异常：{type(exc).__name__}: {exc}")
        if os.getenv("KB_SEED_DEBUG") == "1":
            raise
        say("       需要完整堆栈请设环境变量 KB_SEED_DEBUG=1 后重跑。")
        return 1
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
