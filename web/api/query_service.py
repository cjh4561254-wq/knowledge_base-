"""
查询流程的接口定义
"""
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, RedirectResponse, StreamingResponse

from processor.query_processor.main_graph import KBQueryWorkflow
from processor.query_processor.nodes.node_answer_output import (
    MAX_CONTEXT_CHARS,
    NodeAnswerOutput,
    build_sources,
)
from utils.mongo_history_utils import clear_history, get_recent_messages
from utils.sse_utils import create_sse_queue, SSEEvent, push_to_session, sse_generator
from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, get_task_result, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED, set_task_result, get_done_task_list, reset_task
from utils.auth_utils import auth_status
from web.api.auth_routes import install_auth
from tool.logger import logger
# 1. 创建应用
app = FastAPI(
    title="掌柜智库-查询API",
    description="此文档是掌柜智库查询流程的API接口说明"
)

# 2. 跨域
# 鉴权接入说明（保持本配置不动的理由）：chat.html 由本服务同源提供，页面内
# API_BASE 取 location.protocol//location.host（web/page/chat.html:364，仅 file://
# 打开时才兜底 127.0.0.1:8001），接口调用（/query /stream /history）全部走同源，
# 同源请求不触发 CORS 校验，会话 cookie 由浏览器自动携带，与 allow_credentials
# 无关。因此「通配源 + 不带凭证」的现状不受登录鉴权影响；将来若真的出现
# 跨源页面调用本服务接口，必须改成具体源白名单 + allow_credentials=True。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源
    # allow_credentials 必须为 False：规范要求「通配源 + 携带凭证」的组合直接被浏览器拒绝，
    # 且本项目 fetch/EventSource 都不发 cookie，开着只会埋一个将来跨源带 cookie 时才发现的坑。
    allow_credentials=False,  # 不携带cookie
    allow_methods=["*"],  # 允许的请求方法
    allow_headers=["*"],  # 允许的请求头
)

# 3. 静态页面路由
@app.get("/chat.html")  # 对外访问地址
async def chat():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path / "page" / "chat.html"
    # 如果不存在，抛出404异常
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)


@app.get("/")  # 根路径跳转聊天页（与 import_service 的 "/" 一致；此前无该路由，登录后访问 / 会 404）
async def index():
    return RedirectResponse(url="/chat.html", status_code=307)


# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")  # ...必须填写
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")

@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    :param background_tasks:
    :param request:
    :return:
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # 处理是不是流式返回结果
    is_stream = request.is_stream

    # 本轮进度清零：必须在 update_task_status 之前、且在两条分支共同的入口处。
    # 为什么（实测 bug）：同一个 session_id 连续问第 2 个问题时，task_utils 的
    # running/done/result 三本账里仍留着第 1 问的全部节点，节点又只做「追加」从不整轮
    # 删除，于是第 2 问一上来（甚至走拒答分支只跑了 2 个节点）前端就显示
    # 「阶段进度（已完成7，进行中0，状态：已完成）」，进度条彻底失去意义。
    # 放在分支之前 => 流式与非流式都覆盖；放在 update_task_status 之前 => 紧随其后
    # 那一次 push_queue 推给前端的首帧就是「空清单 + processing」，不会先闪旧清单。
    # 只清三本账，不销毁 SSE 队列（前端 EventSource 可能还连着这个 session），
    # 取舍细节见 utils/task_utils.py::reset_task 的 docstring。
    reset_task(session_id)

    if is_stream:
        # 创建一个字典 存储对一个session_id : queue 结果队列
        create_sse_queue(session_id)
    # 更新任务状态
    # 当前会话id作为key! 整体装填处于运行中！
    update_task_status(session_id, TASK_STATUS_PROCESSING,is_stream)

    print("开始处理流程... 是否流式:", is_stream, f"其他参数:{user_query}, session_id:{session_id}")

    if is_stream:
        # 如果是流式，则返回一个流式响应，过程不断地推送
        # 运行执行图对象方法
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        # 返回结果
        print("开始处理结果....")
        return {
            "message":"结果正在处理中...",
            "session_id":session_id
        }
    else:
        # 同步运行：run_query_graph 在 invoke 模式下返回执行完毕的最终状态，
        # 直接从中取答案/图片/来源，保证非流式响应字段与前端 chat.html 消费处逐条对齐
        # （前端非流式分支读取：data.answer / data.image_urls / data.sources / data.item_names
        #  / data.done_list / data.error）。
        final_state = run_query_graph(session_id, user_query, is_stream)
        # 答案优先取任务结果（节点内 set_task_result 写入），兜底再取最终状态里的 answer
        answer = get_task_result(session_id, "answer", "") or (final_state or {}).get("answer", "")
        # 图片：非流式没有 SSE final 事件携带 image_urls，这里从重排文档中现算，
        # 复用答案输出节点的提取逻辑，避免与流式口径不一致
        image_urls = _extract_image_urls(final_state)
        # 引用来源：非流式同样没有 final 载荷带 sources，缺它前端就只能把答案里的
        # [doc:N] 降级成无标记文本，用户看不到「这段话引自哪几篇」，故在此补齐
        sources = _extract_sources(final_state)
        # 商品名：本轮确认到的主体名，字段名与流式 final 的 item_names 一致
        item_names = (final_state or {}).get("item_names") or []
        # 错误：流程异常时由 run_query_graph 写入任务结果
        error = get_task_result(session_id, "error", "")
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            "image_urls": image_urls,
            # 下面两个字段名与流式 SSE final 事件（node_answer_output.assemble_final_payload）
            # 逐字一致，前端一套渲染代码同时吃流式与非流式
            "item_names": item_names,
            "sources": sources,
            "done_list": get_done_task_list(session_id),
            "error": error,
        }


def _extract_sources(final_state) -> list:
    """
    还原非流式响应的引用来源列表（字段契约见 node_answer_output.build_sources：
    {"index","ref","chunk_id","source","title","url","score"}，不可改名）。

    为什么要重新走一遍 _format_reranked_docs，而不是直接 build_sources(reranked_docs)：
    sources 的 index 必须与 prompt 里的 [N] 编号严格一致，而 prompt 受 MAX_CONTEXT_CHARS
    字符预算约束只收录前若干篇（被预算截掉的文档在 prompt 里没有编号，也不该进来源列表，
    否则前端角标会指向模型根本没看到的资料）。答案节点内部那份 included_docs 既没落进
    state 也没写进 task_result，非流式又没有 SSE final 可以携带，所以这里用同一个方法 +
    同一个预算重算一次，保证编号口径与流式逐条对齐（node_answer_output 属别的包，不动它）。
    """
    if not isinstance(final_state, dict):
        return []
    reranked_docs = final_state.get("reranked_docs") or []
    if not reranked_docs:
        # 空召回 / 追问拒答分支（answer 来自确认节点）本来就没有可引用资料
        return []
    try:
        _context_str, _remain_budget, included_docs = NodeAnswerOutput()._format_reranked_docs(
            list(reranked_docs), MAX_CONTEXT_CHARS
        )
        return build_sources(included_docs)
    except Exception as e:
        # 提取来源失败不能拖垮答案：前端会走「删标记」降级，不会把原始 [doc:N] 漏给用户
        logger.warning(f"非流式响应提取引用来源失败: {e}")
        return []


def _extract_image_urls(final_state) -> list:
    """
    从查询最终状态的重排文档中提取图片 URL（供非流式响应返回）。
    为什么复用 NodeAnswerOutput 的提取方法：与流式 final 事件的 image_urls 保持同一口径，
    前端刷新历史 / 非流式 / 流式三条链路拿到的图片字段结构一致。
    """
    if not isinstance(final_state, dict):
        return []
    try:
        return NodeAnswerOutput()._extract_images_from_docs(final_state.get("reranked_docs") or [])
    except Exception as e:
        # 提取图片失败不应影响答案返回
        logger.warning(f"非流式响应提取图片失败: {e}")
        return []


# 定义查询接口
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    print(f"开始流程图处理...{session_id} {user_query} {is_stream}")

    # 兜底清零（与 /query 入口那次是同一语义，重复调用幂等）：
    # 为什么两处都清：/query 的清零保证「前端首帧就是空清单」；本处的清零保证
    # 「无论谁调用这个函数（非流式 BackgroundTasks、直接单测、将来的其它入口），
    # 本轮进度都从空开始」——这才是 bug 的根因位置：账本 key 是 session_id，跨轮复用。
    # 安全性：清零点在这里仍然「早于本轮任何节点打点」（workflow.run 尚未开始 /
    # 生成器尚未被消费），所以不会把本轮已经跑出来的进度抹掉；
    # 若上一轮仍在同会话后台执行，被清的只是它的展示账本，节点本身照常执行并会
    # 重新 add_running_task/add_done_task 回来（追加是幂等的），不抛异常。
    # reset_status=False：状态已由 /query 置为 processing，这里不回退成 pending，
    # 否则流式期间前端会一直显示「等待中」，把上一轮修好的状态标签又改坏。
    reset_task(session_id, reset_status=False)

    init_state = {
        "original_query": user_query,
        "session_id": session_id,
        "is_stream": is_stream
    }

    final_state = None
    try:
        workflow = KBQueryWorkflow()
        if is_stream:
            # 流式：迭代图执行产生的分片（节点进度/final 等由各节点自行 push 到 SSE 队列）
            for chunk in workflow.run(init_state, stream=True):
                logger.debug(chunk)
        else:
            # 非流式：invoke 直接返回最终状态，供 /query 组装响应
            final_state = workflow.run(init_state, stream=False)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        print(f"流程执行异常: {e}")
        # 记录错误到任务结果，供非流式 /query 返回 data.error；流式则通过 SSE error 事件下发
        set_task_result(session_id, "error", str(e))
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
    finally:
        # 无论成功/异常，流式都在最后补一个 CLOSE：
        # 为什么：图节点只 push final，一旦 final 丢失（异常、提前结束），前端 EventSource
        # 会永久挂住。CLOSE 触发 sse_generator 补发 done 终止帧并结束流，给客户端兜底。
        # CLOSE 在 final 之后到达时，前端已 es.close()，此帧会被安全忽略。
        if is_stream:
            push_to_session(session_id, SSEEvent.CLOSE, {})

    return final_state



@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):

    print("调用流式/stream...")
    """
    sse 实时返回结果
    """
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    清空指定会话的历史记录
    """
    count = clear_history(session_id)
    return {"message": "历史会话已清空", "deleted_count": count}

@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                # 补齐 image_urls：前端 loadHistory 用 item.image_urls 渲染图片，
                # 缺这个字段刷新历史后引用图片会丢失（mongo_history_utils 落库时 None 已规范为 []）
                "image_urls": r.get("image_urls", []),
                # sources 透传：库里目前没有这个字段（save_chat_message 不落 sources，属 utils 包，未擅自改），
                # 取不到就返回 [] —— 前端据此走「删掉 [doc:N] 标记」的降级，绝不会把原始标记显示给用户。
                # 等主控决定在 mongo_history_utils 落 sources 后，这里不用改代码即可还原引用角标。
                "sources": r.get("sources") or [],
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


# 证明服务器启动即可
@app.get("/health")
async def health():
    """
    检查服务是否正常
    """
    # 保持不鉴权（install_auth 白名单放行 /health，运维探测要用）。
    # 附鉴权子系统摘要：只取 mongo/users 两个计数字段，刻意不带 detail ——
    # auth_status() 失败分支的 detail 文案里含 MONGO_URL，可能带库的凭据。
    status = auth_status()
    return {"ok": True, "auth": {"mongo": status.get("mongo"), "users": status.get("users")}}

# 4. 登录鉴权接入（实现在 web/api/auth_routes.py，此处只按契约挂载）：
#    - 未登录访问 page_routes 里的页面 → 302 /login.html?next=...
#    - 未登录调用 api_prefixes 前缀的接口 → 401 {"ok":false,"error":"未登录或会话已过期"}，
#      请求在中间件层被挡掉，业务逻辑（/query 的 LangGraph、/history 的 Mongo 读写、
#      /stream 的 SSE 队列）根本不会执行。
#    - 放行 /login.html /login /logout /api/me /health /docs /openapi.json /redoc。
#    页面绝对路径用 __file__ 锚定到项目根（不依赖启动时的 cwd）。
#    下方 install_auth 之后，原有各业务路由函数体一律未改动。
_PAGE_DIR = Path(__file__).resolve().parents[2] / "web" / "page"
install_auth(
    app,
    page_routes={
        "/chat.html": _PAGE_DIR / "chat.html",
        "/": _PAGE_DIR / "chat.html",
    },
    api_prefixes=("/query", "/stream", "/history"),
)

if __name__ == "__main__":
    # 注意：本项目必须从项目根用 -m 方式启动（`python -m uvicorn web.api.query_service:app --port 8001`），
    # 直接运行本文件可能因 sys.path 不含项目根而报 No module named 'config'。
    uvicorn.run(app, host="127.0.0.1", port=8001)