# web/api/auth_routes.py
"""
登录鉴权路由层（导入服务 8000 / 检索服务 8001 共用）
====================================================

定位：`utils/auth_utils.py` 只提供核心能力（散列/会话/限流/依赖），本模块把它组装成
可直接挂到任意 FastAPI app 上的两层东西：

1. ``router`` —— APIRouter，含 5 个路由：
   * POST /login            登录（cookie 会话），契约与 web/page/login.html 对齐：
                            200 {ok:true, username, display_name}
                            400/401/429/503 {ok:false, error:"中文可读文案"}
   * POST /logout           注销：delete_session + 同一个响应对象上 delete_cookie；幂等
   * GET  /api/me           当前身份（登录页加载即探测用）；200 {username, display_name}
   * GET  /login.html       登录页 FileResponse（Path(__file__) 锚定，不依赖 cwd）
   * GET  /api/auth/status  排障探针：auth_utils.auth_status()，只有计数没有密码字段

2. ``install_auth(app, page_routes=..., api_prefixes=...)`` —— 挂 router + 一个 http 中间件：
   * 白名单直接放行（登录页/登录/注销/me/health/docs 系/favicon）
   * 其余请求解析会话（cookie 优先，其次 Authorization: Bearer —— 后者是给 curl /
     自动化脚本留的排障通道，auth_utils.read_session_token 已内置，中间件复用即可）
   * 有效会话 → request.state.user = user dict，业务路由可直接取，不必再挂依赖
   * 无有效会话：命中 page_routes 的 GET → 302 /login.html?next=<urlencoded path>；
     其它一切（api_prefixes 或任何非白名单路径）→ 401 {"ok":false,"error":...}
   * AuthStorageError（Mongo 挂）→ 503 带可读文案，绝不 500 崩栈

对接方式（两个服务各自在 import 区后加三行，页面/业务路由本身不动）：
    from web.api.auth_routes import install_auth
    install_auth(
        app,
        page_routes={"/chat.html": str(Path(__file__).resolve().parent.parent / "page" / "chat.html"),
                     "/":        str(Path(__file__).resolve().parent.parent / "page" / "chat.html")},
        api_prefixes=("/query", "/stream", "/history"),
    )

注意（CORS 顺序）：本模块的中间件在 install_auth 里会被刻意挪到 user_middleware 末尾
（最内层），确保服务里先注册的 CORSMiddleware 在外面包裹它 —— 否则未登录的 401/302
响应不带 Access-Control-Allow-* 头，跨源前端（8001 页面调 8000 接口）会把 401 读成
"网络错误"，登录过期提示就永远弹不出来。
"""

import math
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

import utils.auth_utils as au
from tool.logger import logger

# 登录页磁盘位置：web/api/auth_routes.py -> parents[2] 是项目根 D:\knowledge_base
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGIN_PAGE_PATH = _PROJECT_ROOT / "web" / "page" / "login.html"

# 中间件放行清单（精确匹配路径，不含 query）：
#   /login.html /login /logout /api/me —— 鉴权自身
#   /api/auth/status —— 排障探针（auth_status 只回计数，无任何密码字段，公开安全）
#   /health /docs /openapi.json /redoc /favicon.ico —— 运维与 Swagger 可用性问题
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/login.html",
        "/login",
        "/logout",
        "/api/me",
        "/api/auth/status",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/favicon.ico",
    }
)

#: 未登录访问业务接口时的统一文案（前端按 status===401 弹"重新登录"）
UNAUTHORIZED_MESSAGE = "未登录或会话已过期"

router = APIRouter(tags=["auth"])


def _fail(status_code: int, error: str) -> JSONResponse:
    """统一的失败响应形态 {"ok": false, "error": "..."}（login.html 按 body.error 显示）。"""
    return JSONResponse(status_code=status_code, content={"ok": False, "error": error})


# ====================
# 路由
# ====================


@router.get("/login.html", summary="登录页")
async def login_page():
    """返回登录页静态文件。

    已登录的人打到这里也照样返回该页：login.html 自带 GET /api/me 探测，
    登录态访问会由前端自己按 ?next= 跳走，服务端不需要在这里做二次判断
    （判断了反而把"登录页必须在白名单里可匿名访问"的简单模型复杂化）。
    """
    if not LOGIN_PAGE_PATH.is_file():
        raise HTTPException(status_code=404, detail=f"登录页不存在：{LOGIN_PAGE_PATH}")
    return FileResponse(LOGIN_PAGE_PATH)


@router.post("/login", summary="登录")
async def login(request: Request):
    """用户名+口令换会话 cookie。

    响应矩阵（与 web/page/login.html 的 interpretLoginResponse 严格对齐）：
      200 {"ok":true,"username","display_name"} + Set-Cookie kb_session
      400 {"ok":false,"error":"…"}             请求体非法 JSON / 字段缺失
      401 {"ok":false,"error":"用户名或密码错误"}  任何凭据级失败（不区分，防枚举）
      429 {"ok":false,"error":"…N分钟后再试"} + Retry-After  命中限流
      503 {"ok":false,"error":"…"}             鉴权存储（Mongo）不可用
    """
    try:
        payload = await request.json()
    except Exception:
        # 非法 JSON / 空 body / 非 UTF-8：都是客户端请求问题，400 说清楚，不抛栈
        return _fail(400, "请求体必须是合法的 JSON")
    if not isinstance(payload, dict):
        return _fail(400, "请求体必须是 JSON 对象（形如 {\"username\":..., \"password\":...}）")

    username = payload.get("username")
    password = payload.get("password")
    if not isinstance(username, str) or not username.strip():
        return _fail(400, "缺少 username 字段（或为空）")
    if not isinstance(password, str) or not password:
        return _fail(400, "缺少 password 字段（或为空）")
    remember = bool(payload.get("remember"))

    try:
        user = au.authenticate(username, password)
    except au.TooManyLoginAttemptsError as e:
        # authenticate 的限流分支是「抛」不是「返回 None」（auth_utils 契约），必须单独接。
        # Retry-After 用 login_throttle_remaining 的真实剩余秒数；探针返回 0 时兜底给整窗。
        remaining = au.login_throttle_remaining(username)
        # remaining 探到 0 只可能是抛错与探测之间窗口刚好到期的竞态，兜底给整窗 15 分钟
        # （刻意写字面量而不是引用 au._LOGIN_WINDOW_SECONDS 私有常量）
        retry_after = int(math.ceil(remaining)) if remaining > 0 else 15 * 60
        response = _fail(429, str(e))
        response.headers["Retry-After"] = str(max(1, retry_after))
        logger.warning(f"[auth-routes] 登录被限流：username={username!r} retry_after={retry_after}s")
        return response
    except au.AuthStorageError as e:
        # 存储故障绝不伪装成"密码错误"
        logger.error(f"[auth-routes] 登录时鉴权存储不可用：{e}")
        return _fail(503, f"登录服务暂时不可用（鉴权存储不可达），请稍后重试：{e}")
    except au.AuthError as e:
        # 其余 AuthError（用户名不合规则等）：回与错密码完全相同的 401 文案，
        # 否则"这个用户名太短/含非法字符"就成了新的枚举信道；细节只进日志。
        logger.warning(f"[auth-routes] 登录参数不合规：type={type(e).__name__} username={username!r}")
        return _fail(401, au.BAD_CREDENTIALS_MESSAGE)

    if user is None:
        # 用户不存在 / 密码错 / 账号禁用：auth_utils 已抹平差异，这里只转发同一句文案
        return _fail(401, au.BAD_CREDENTIALS_MESSAGE)

    try:
        token, _session = au.create_session(user["_id"], remember=remember)
    except au.AuthStorageError as e:
        logger.error(f"[auth-routes] 建会话时鉴权存储不可用：{e}")
        return _fail(503, f"登录服务暂时不可用（建会话失败）：{e}")
    except au.AuthError as e:
        # 理论上到不了（authenticate 刚返回的用户必然存在）；真到了也不能崩栈
        logger.error(f"[auth-routes] 建会话异常：{e}")
        return _fail(401, "登录异常，请重试")

    response = JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "username": user.get("username", ""),
            "display_name": user.get("display_name") or user.get("username", ""),
        },
    )
    # cookie flag（HttpOnly / SameSite=lax / Path=/ / Secure<-AUTH_COOKIE_SECURE / max_age
    # 按 remember 取 12h 或 30d）全部收敛在 auth_utils.set_session_cookie 一处实现，
    # 不在这里手抄一遍，避免两处拼错。
    au.set_session_cookie(response, token, remember=remember)
    # 日志纪律：只记用户名与结果，绝不记 password / token / 散列
    logger.info(f"[auth-routes] 登录成功：username={user.get('username')!r} remember={remember}")
    return response


@router.post("/logout", summary="注销")
async def logout(request: Request):
    """删除服务端会话 + 清浏览器 cookie。

    幂等：未登录（无 token）调用 delete_session(None) 返回 False 且不抛，照样 200。
    两步都必须做：只 delete_cookie 是"浏览器自己忘掉"，库里的记录还能继续过校验。
    delete_cookie 作用在**本响应的同一个对象**上，否则头不会跟着这次响应出去。
    """
    token = au.read_session_token(request)
    storage_error: str | None = None
    try:
        deleted = au.delete_session(token)
    except au.AuthStorageError as e:
        storage_error = str(e)
        deleted = False
        logger.error(f"[auth-routes] 注销时鉴权存储不可用：{e}")

    if storage_error is None:
        response = JSONResponse(status_code=200, content={"ok": True})
    else:
        # 旧会话没删掉 = 凭据仍有效，必须让用户知道（同时仍清 cookie，能救一半是一半）
        response = _fail(503, f"注销未完全成功（登录存储不可用），请登录设备后在浏览器清除站点数据：{storage_error}")
    au.clear_session_cookie(response)
    logger.info(f"[auth-routes] 注销：deleted={deleted} has_token={bool(token)}")
    return response


@router.get("/api/me", summary="当前登录身份")
async def me(request: Request):
    """登录页加载即探测：200 {ok:true, username, display_name} / 401 / 503。

    在白名单里（不靠中间件拦），因为 login.html 打开第一件事就是调它，
    401 必须是"可读的 JSON 状态"而不是中间件的通用拦截——两者形态恰好一致，
    这里自己解析会话并把 503 与 401 分开即可。
    """
    token = au.read_session_token(request)
    try:
        sess = au.get_session(token) if token else None
    except au.AuthStorageError as e:
        return _fail(503, f"登录服务暂时不可用（鉴权存储不可达）：{e}")
    if sess is None:
        return _fail(401, UNAUTHORIZED_MESSAGE)
    user = sess["user"]
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "username": user.get("username", ""),
            "display_name": user.get("display_name") or user.get("username", ""),
        },
    )


@router.get("/api/auth/status", summary="鉴权子系统状态（排障）")
async def auth_status_probe():
    """auth_utils.auth_status() 原样透出：{mongo, users, detail}。

    auth_status 内部只 count，不返回任何用户文档，绝无密码字段；且它契约就是
    "不抛异常"，Mongo 挂时给出结论而不是 500，所以本路由不需要 try。
    """
    return au.auth_status()


# ====================
# 中间件安装
# ====================


def install_auth(
    app: FastAPI,
    *,
    page_routes: dict[str, str],
    api_prefixes: tuple[str, ...],
) -> None:
    """把登录鉴权装进一个 FastAPI app（路由 + 门禁中间件），一行接入。

    参数
    ----
    page_routes:
        {受保护页面路径: 页面磁盘绝对路径}，例如
        {"/chat.html": r"D:\\knowledge_base\\web\\page\\chat.html", "/": 同上}。
        未登录 GET 命中这些路径 → 302 /login.html?next=<urlencoded 原始 path>。
        （值当下只用作"这是个页面"的判定依据；返回页面本身仍由服务自己的
        FileResponse 路由负责——中间件放行后自然会走到那里。）
    api_prefixes:
        需要登录的 API 路径前缀元组（如 ("/query", "/stream", "/history")）。
        实现上任何非白名单路径未登录都是一律 401（见模块 docstring 的安全默认策略），
        所以该参数不改变拦截结果，保留它是为了：意图显式声明 + 将来要做
        "前缀级豁免（如公开搜索）"时只改这里一处。

    行为
    ----
    1) app.include_router(router)
    2) 注册 http 中间件：白名单放行；有会话则 request.state.user 注入并放行
       （顺带 touch_session 做滑动续期，auth_utils 内部 60s 节流 + 永不抛）；
       无会话按上面规则 302 / 401；AuthStorageError → 503 JSON，不崩栈。
    3) 把该中间件挪到 user_middleware 末尾（最内层），保证服务先注册的
       CORSMiddleware 包在它外面 —— 见模块 docstring 的 CORS 顺序说明。
    """
    pages = dict(page_routes or {})
    prefixes = tuple(api_prefixes or ())

    app.include_router(router)

    async def auth_gate(request: Request, call_next):
        path = request.url.path

        # CORS 预检不带 cookie、也不该关心登录态：放行交给内层 CORSMiddleware 应答。
        # （即便本中间件被挪到 CORS 外面运行，这一条也保证 preflight 绝不被 401。）
        if request.method == "OPTIONS":
            return await call_next(request)
        if path in PUBLIC_PATHS:
            return await call_next(request)

        token = au.read_session_token(request)  # cookie 优先，其次 Authorization: Bearer
        try:
            sess = au.get_session(token) if token else None
        except au.AuthStorageError as e:
            logger.error(f"[auth-gate] 会话解析时鉴权存储不可用：{e}")
            return _fail(503, f"登录服务暂时不可用（鉴权存储不可达），请稍后重试：{e}")

        if sess is not None:
            request.state.user = sess["user"]
            request.state.session = sess["session"]
            # 滑动续期：get_session 不写库，这里补一次带 60s 内存节流的机会；
            # touch_session 契约是永不抛，续期失败不该把已登录用户踢下线。
            au.touch_session(token)
            return await call_next(request)

        # —— 以下都是"没有有效会话" ——
        if path in pages and request.method == "GET":
            nxt = quote(path, safe="")
            return RedirectResponse(status_code=302, url=f"{au.LOGIN_PATH}?next={nxt}")
        # api_prefixes 命中 或 任何非白名单未知路径：默认拒绝（fail-closed）。
        # 未知路径也 401 是刻意的：新增接口忘了登记前缀时的后果应该是"进不去"，
        # 而不是"裸奔"。
        _ = prefixes
        return _fail(401, UNAUTHORIZED_MESSAGE)

    app.add_middleware(BaseHTTPMiddleware, dispatch=auth_gate)
    # add_middleware 是插到列表头部（越靠后注册越外层），这里把它挪到末尾，
    # 让先注册的 CORSMiddleware 成为最外层，保证 401/302/503 响应也带 CORS 头。
    if app.user_middleware and app.user_middleware[0].cls is BaseHTTPMiddleware:
        app.user_middleware.append(app.user_middleware.pop(0))
    logger.info(
        f"[auth-routes] 鉴权已安装：pages={sorted(pages)} prefixes={sorted(prefixes)} "
        f"public={sorted(PUBLIC_PATHS)}"
    )
