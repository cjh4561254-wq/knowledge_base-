# web/api/import_service.py
"""
BOSS智库 · 导入侧 Web 服务

职责边界（与检索侧 web/api/query_service.py 对称）：
    * 只提供「上传 → 触发后台导入图 → 查进度 → 取产物」这一条 HTTP 通道，
      业务逻辑全在 processor/import_processor 的 LangGraph 节点里。
    * 服务直接由 uvicorn 挂载本模块的 app 启动（`uvicorn web.api.import_service:app`），
      项目根的 main.py 是空文件，**不要**依赖它做启动入口。

进度获取方式（重要，前后端必须一致）：
    导入侧用**轮询** `GET /status/{task_id}`（课程 day13/19 的原始设计），字段来自
    utils/task_utils.py 的内存任务表：status / done_list / running_list（中英名字典见
    task_utils._NODE_NAME_TO_CN，接口返回的是**中文展示名**）。
    同时本模块保留 SSE 兼容路径：若有人为 task_id 预先 create_sse_queue，
    失败时会补发 error + __close__ 事件（事件名取自 utils/sse_utils.SSEEvent），
    这样导入进度也能用 EventSource 订阅，两条通道字段完全同名。
"""

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from config.minio_config import minio_config
from processor.import_processor.exceptions import ImportProcessError
from processor.import_processor.main_graph import KBImportWorkflow
from tool.logger import logger
from utils.auth_utils import auth_status
from utils.minio_utils import get_minio_client
from utils.sse_utils import SSEEvent, get_sse_queue, push_to_session
from web.api.auth_routes import install_auth
from utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PENDING,
    TASK_STATUS_PROCESSING,
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)

# 1. 创建应用
# 标题和描述会在Swagger文档中展示
app = FastAPI(
    title="BOSS智库-导入API",
    description="此文档是BOSS智库导入流程的API接口说明"
)

# 允许导入的文件后缀：必须与 processor/import_processor/nodes/node_entry._TYPE_ROUTES 保持一致，
# 否则这里放行的文件会在入口节点被判「不支持的类型」后静默走到 END（任务显示完成却什么都没导入）。
ALLOWED_EXTENSIONS = (".pdf", ".md")

# 跨域允许的来源：覆盖前端所有可能的打开方式
#   - http://localhost:8000 / http://127.0.0.1:8000：本服务自托管 import.html（课程默认用法）
#   - http://localhost:5173 / http://127.0.0.1:5173：前端独立 dev server
#   - http://127.0.0.1:8001：检索服务页面也要调本服务的接口
# 为什么不用 allow_origins=["*"] 配 allow_credentials=True：
#   该组合下浏览器会拒绝 `Access-Control-Allow-Origin: *` 的带凭证响应（CORS 规范），
#   而且文档里特别提醒「localhost 与 127.0.0.1 也算跨域」，硬写一个源迟早踩坑。
#   用「具体源列表 + 正则」让中间件回显真实 Origin，credentials 才合法可用。
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]


def _cors_origins() -> List[str]:
    """允许用环境变量 CORS_ALLOW_ORIGINS 追加来源（逗号分隔），不改代码就能加源。"""
    extra = os.getenv("CORS_ALLOW_ORIGINS", "")
    origins = list(_DEFAULT_CORS_ORIGINS)
    origins.extend(item.strip() for item in extra.split(",") if item.strip())
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    # 兜住任意本机端口（如 vite 落到 5174）：正则命中时 Starlette 会回显具体 Origin
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["*"],  # 允许的请求方法
    allow_headers=["*"],  # 允许的请求头
)


def _fix_mojibake_filename(name: str) -> str:
    """
    修复 multipart 上传的中文文件名乱码。

    为什么必须做：客户端不带 RFC 5987 的 `filename*` 时，starlette/python-multipart
    按 latin-1 解 Content-Disposition，于是多字节文件名的原始字节被当成 latin-1 字符。
    本机实测到两种真实形态，所以按顺序试两种解码目标：
      - 浏览器（UTF-8 字节）：`03-商家...` 变成 U+00E5 U+0095 U+0086…  → latin-1 → utf-8 可还原
      - Windows curl（GBK 字节）：`03-商家...` 变成 U+00C9 U+00CC U+00BC… → latin-1 → gbk 可还原
    直接后果曾实测：同一份文档在 Milvus 里落下两套 file_title（一份正常一份乱码），
    检索出重复内容，MinIO 图片前缀也被劈成两个目录。

    只在「能 latin-1 回编 && 能按目标编码解出 && 结果确实不同」时才替换，
    纯 ASCII 与真正的 latin-1 名（如 café.pdf）都会原样返回。
    """
    try:
        raw = name.encode("latin-1")
    except UnicodeEncodeError:
        return name
    for enc in ("utf-8", "gbk"):
        try:
            repaired = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if repaired != name:
            return repaired
    return name


def _safe_filename(raw_filename: str | None) -> str:
    """
    清洗上传文件名，只保留纯文件名。

    为什么必须做：浏览器给的 file.filename 可能是 "D:\\x\\y.pdf" 或
    "..\\..\\windows\\system32\\x.pdf"（IE/部分客户端会带全路径），
    直接 os.path.join 到 file_dir 就写成目录穿越了。
    不用 werkzeug secure_filename：它会把中文文件名整个丢掉，本项目全是中文手册名。
    """
    name = (raw_filename or "").replace("\\", "/").split("/")[-1].strip()
    name = _fix_mojibake_filename(name)
    # 去掉盘符残留（"C:x.pdf" 这种）与控制字符
    name = name.lstrip(".").strip()
    return name


def _data_root_dir() -> str:
    """取数据根目录（缺失时给出可读 500，而不是 os.path.join(None) 的 TypeError）。"""
    root = (os.getenv("DATA_BASED_ROOT_DIR") or "").strip()
    if not root:
        raise HTTPException(
            status_code=500,
            detail="服务端未配置数据根目录 DATA_BASED_ROOT_DIR，无法缓存上传文件（请检查 .env）",
        )
    return root


# 3. 静态页面路由：返回文件导入前端页面
# 访问地址：http://localhost:8000/import.html
@app.get("/import.html")  # 对外访问地址
async def get_import_page():
    # 拼接HTML文件绝对路径
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path / "page" / "import.html"
    # 如果不存在，抛出404异常
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)


@app.get("/", summary="根路径跳转导入页")
async def index():
    """让浏览器直接访问 http://localhost:8000/ 就能进导入页（省得记 .html）。"""
    return RedirectResponse(url="/import.html", status_code=307)


@app.get("/health", summary="健康检查")
async def health():
    """存活探针：主控串行验证时先 curl 这个，确认服务起来了再测上传。"""
    # 保持不鉴权（install_auth 白名单放行 /health，运维探测要用）。
    # 附鉴权子系统摘要：只取 mongo/users 两个计数字段，刻意不带 detail ——
    # auth_status() 失败分支的 detail 文案里含 MONGO_URL，可能带库的凭据。
    status = auth_status()
    return {
        "ok": True,
        "service": "import",
        "auth": {"mongo": status.get("mongo"), "users": status.get("users")},
    }


# 4. 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
def _publish_failure(task_id: str, error_info: Dict[str, Any]) -> None:
    """
    失败信息落地：写内存任务表 + （若有人在监听）补发 SSE 错误事件。

    为什么两边都发：导入侧前端目前用轮询（/status），错误必须能从任务表读到；
    但 utils/sse_utils 的错误事件通道已经存在，将来进度改 SSE 时不用回来补代码。
    只在队列真实存在时才 push：sse_utils 的 push_to_session 对缺失队列会打 warning，
    轮询场景下每个失败任务都会白刷一条日志。
    """
    set_task_result(task_id, "error", error_info.get("message", "") or "导入流程执行失败")
    if get_sse_queue(task_id) is not None:
        push_to_session(task_id, SSEEvent.ERROR, error_info)
        # CLOSE 是内部信号，sse_generator 收到后会转成 done 终止帧，避免客户端挂流等待
        push_to_session(task_id, SSEEvent.CLOSE, {})


def _record_result(task_id: str, final_state: Dict[str, Any]) -> None:
    """把关键产物路径/统计写进任务结果表，供 /status 与 /result 接口回查。"""
    chunks = final_state.get("chunks") or []
    set_task_result(task_id, "md_path", str(final_state.get("md_path", "") or ""))
    set_task_result(task_id, "file_title", str(final_state.get("file_title", "") or ""))
    set_task_result(task_id, "item_name", str(final_state.get("item_name", "") or ""))
    set_task_result(task_id, "chunk_count", str(len(chunks)))


def run_graph_task(task_id: str, file_dir: str, import_file_path: str):
    """
    LangGraph全流程执行后台任务
    核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看

    :param task_id: 全局唯一任务ID，关联单个文件的全流程处理
    :param file_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
    :param import_file_path: 上传文件的本地绝对路径
    """
    # 只留最新一次节点增量：main_graph 用 stream_mode="updates"，
    # 节点返回的是整份 state，所以最后一个事件足以还原最终产物路径。
    latest_state: Dict[str, Any] = {}
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        # 2. 初始化LangGraph状态
        init_state = {
            "task_id": task_id,  # 任务ID
            "file_dir": file_dir,  # 输出路径
            "import_file_path": import_file_path,  # 导入路径
        }

        # 3. 流式执行LangGraph全流程（stream模式：实时获取每个节点的执行结果）
        workflow = KBImportWorkflow()  # 创建工作流 调你整个 导入流程的 工作流
        for event in workflow.run(init_state, stream=True):  # 让整个导入工作流 跑起来 流式输出
            for node_name, node_result in event.items():
                # 将完成的节点名加入【已完成列表】，前端轮询/status/{task_id}可实时获取
                # 注：节点内部 BaseNode.__call__ 已经打过点，这里 add_done_task 幂等去重，
                # 保留是为了覆盖「图级路由节点/旁路节点」这类没挂 BaseNode 的增量。
                add_done_task(task_id, node_name)
                if isinstance(node_result, dict):
                    latest_state.update(node_result)

        # 4. 全流程执行完成，更新任务全局状态为：已完成
        _record_result(task_id, latest_state)
        update_task_status(task_id, TASK_STATUS_COMPLETED)

    except ImportProcessError as e:
        # 5. 节点级异常：main_graph 已 set_task_result("error") 并原样上抛，
        #    这里负责把**结构化错误**（含出错节点名）落到任务表并收尾状态，
        #    绝不能再退化成一句 logger.info —— 那样前端只能看到「失败」却不知道为什么。
        error_info = e.to_dict()
        logger.error(
            f"[{task_id}] LangGraph全流程执行失败 出错节点={error_info.get('node_name') or '-'} "
            f"信息：{error_info.get('message')}",
            exc_info=True,
        )
        _publish_failure(task_id, error_info)
        # 出错节点已被 BaseNode 从 running 移到 done，这里显式定格失败态
        update_task_status(task_id, TASK_STATUS_FAILED)

    except Exception as e:  # 图之外的异常（构图/状态/依赖缺失）也要给出可读原因
        logger.error(f"[{task_id}] LangGraph全流程执行失败（非节点异常）：{e}", exc_info=True)
        error_info = {
            "error_type": type(e).__name__,
            "node_name": "import_graph",
            "message": f"导入流程异常：{e}",
            "cause": str(e),
        }
        _publish_failure(task_id, error_info)
        update_task_status(task_id, TASK_STATUS_FAILED)


# 5. 核心接口：文件上传接口
# 支持多文件上传，核心流程：接收文件 → 本地保存 → MinIO上传 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参，字段名 files）
@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    文件上传核心接口
    1. 校验并清洗文件名（防目录穿越、只放行 PDF/MD）
    2. 按「日期/任务ID」分层保存到本地中转目录，避免文件冲突
    3. 将文件上传至MinIO对象存储，做持久化保存（失败即报错，不放行）
    4. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    5. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式，字段名必须是 files）
    :return: 包含上传结果和所有任务ID的JSON响应
    :raises HTTPException: 400 文件不合法 / 500 本地落盘失败 / 502 对象存储不可用
    """
    # 1. 构建本地存储根目录：数据根目录/YYYYMMDD（按日期分层，方便管理）
    data_based_root_dir = _data_root_dir()
    data_dir = os.path.join(data_based_root_dir, datetime.now().strftime("%Y%m%d"))
    # 初始化任务ID列表，用于返回给前端（一个文件对应一个TaskID）
    task_ids: List[str] = []
    items: List[Dict[str, Any]] = []

    if not files:
        raise HTTPException(status_code=400, detail="未接收到任何文件（form-data 字段名应为 files）")

    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成TaskID）
    for file in files:
        # 生成全局唯一TaskID（UUID4），作为单个文件的全流程标识
        task_id = str(uuid.uuid4())  # 会为每一个 PDF上传的流程 生成唯一的任务ID （追踪）

        # 2.1 文件名清洗 + 类型校验：把「不支持的类型 / 非法文件名」挡在图执行之前
        #     为什么要挡：node_entry 遇到非法后缀会直接路由到 END，任务最终状态仍是
        #     completed，前端看到「已完成」但库里什么都没写，属于最难排查的一类假成功。
        safe_name = _safe_filename(file.filename)
        if not safe_name:
            raise HTTPException(status_code=400, detail="上传文件名为空或不合法，请重新选择文件")
        if os.path.splitext(safe_name)[1].lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"文件「{safe_name}」类型不支持，仅支持 {'/'.join(ALLOWED_EXTENSIONS)} 格式",
            )

        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{safe_name}，文件类型：{file.content_type}")

        # 2.2 先把状态置为 pending：前端可能在后台任务启动前就发起第一次轮询，
        #     此时 get_task_status 返回空串会被当成「未知任务」而 404。
        update_task_status(task_id, TASK_STATUS_PENDING)
        # 标记「文件上传」阶段为「运行中」，前端轮询可查   添加一个标记
        add_running_task(task_id, "upload_file")

        try:
            # 3. 构建该任务的本地独立目录：数据根目录/YYYYMMDD/TaskID，避免多文件重名冲突
            file_dir = os.path.join(data_dir, task_id)
            os.makedirs(file_dir, exist_ok=True)  # 目录不存在则创建，存在则不做处理
            # 构建上传文件的本地保存绝对路径
            import_file_path = os.path.join(file_dir, safe_name)

            # 4. 将上传的文件保存到本地中转目录（后续MinIO上传/文件解析均基于此文件）
            with open(import_file_path, "wb") as file_buffer:
                shutil.copyfileobj(file.file, file_buffer)  # 复制
            logger.info(f"[{task_id}] 文件已保存至本地，路径：{import_file_path}")

            # 5. 将本地文件上传至MinIO对象存储，做持久化保存
            # 构建MinIO中的文件对象名：pdf_files/YYYYMMDD/文件名（按日期分层，和本地一致） MINIO上的路径
            minio_object_name = f"pdf_files/{datetime.now().strftime('%Y%m%d')}/{safe_name}"
            try:
                minio_client = get_minio_client()
                minio_bucket_name = minio_config.bucket_name
                minio_client.fput_object(
                    bucket_name=minio_bucket_name,  # 桶
                    object_name=minio_object_name,  # pdf_files/20260728/H3C xx手册.pdf
                    file_path=import_file_path,  # D:\file\20260728\<uuid>\xx.pdf
                    content_type=file.content_type,  # 传递文件原始MIME类型
                )
                logger.info(
                    f"[{task_id}] 文件已成功上传至MinIO，桶名：{minio_bucket_name}，对象名：{minio_object_name}"
                )
            except Exception as e:
                # 【关键修复】对象存储不可用必须**中断**本次上传并明确报错，
                # 旧版只 logger.warning 后继续，后果是：
                #   1) 调用方拿到 200 + task_id，以为入库成功，实际中转链路是断的；
                #   2) node_md_img 后续要往同一套 MinIO 写图片，必挂在更深的节点上，
                #      错误信息还被包成「图片处理失败」，根因（MinIO 连不上）看不出来。
                # 用 502：语义是「上游存储服务不可用」，与 500（本服务自己的错）区分开，
                # 运维一眼知道去查 localhost:9100 而不是查代码。
                logger.error(
                    f"[{task_id}] 文件上传MinIO失败，已中断导入流程，异常信息：{e}", exc_info=True
                )
                set_task_result(task_id, "error", f"文件转存对象存储失败：{e}")
                update_task_status(task_id, TASK_STATUS_FAILED)
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"文件「{safe_name}」转存对象存储（MinIO {minio_config.endpoint}）失败："
                        f"{e}；请确认 MinIO 服务与桶 {minio_config.bucket_name} 可用后重试"
                    ),
                ) from e

            # 6. 标记「文件上传」阶段为「已完成」，前端轮询可查
            add_done_task(task_id, "upload_file")

            # 7. 将LangGraph全流程处理加入FastAPI后台任务（异步执行，不阻塞当前接口响应）
            background_tasks.add_task(run_graph_task, task_id, file_dir, import_file_path)
            logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

            items.append(
                {
                    # 导入侧 task_id 即会话 key（task_push_queue/SSE 都用它），故同时返回 session_id
                    "task_id": task_id,
                    "session_id": task_id,
                    "filename": safe_name,
                    "file_dir": file_dir,
                    "import_file_path": import_file_path,
                }
            )
        except HTTPException:
            # 上面主动抛的 4xx/5xx 原样冒泡，别让下面的兜底把它压成 500
            raise
        except Exception as e:
            # 本地落盘/建目录失败（磁盘满、权限、DATA_BASED_ROOT_DIR 不可写）
            logger.error(f"[{task_id}] 文件保存到本地中转目录失败：{e}", exc_info=True)
            set_task_result(task_id, "error", f"文件保存失败：{e}")
            update_task_status(task_id, TASK_STATUS_FAILED)
            raise HTTPException(
                status_code=500,
                detail=f"文件「{safe_name}」保存到服务器失败：{e}；请检查 DATA_BASED_ROOT_DIR 是否可写",
            ) from e

    # 8. 所有文件处理完毕，返回上传成功信息和所有TaskID（前端基于TaskID轮询进度）
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {
        "code": 200,
        "message": f" 文件上传成功, total: {len(files)}",
        "task_ids": task_ids,
        "items": items,
    }


# 6. 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每2秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点、失败原因、产物信息的JSON响应
    """
    status = get_task_status(task_id)
    # 未知任务：前端必须先停止轮询，否则每2秒空转到服务重启都得不到答案
    # （旧版返回 status:"" 时前端既不 completed 也不 failed，会无限轮询且界面毫无提示）
    if status == "":
        raise HTTPException(
            status_code=404,
            detail=f"未找到任务 {task_id}（可能已过期或服务已重启），请重新上传文件",
        )

    # 任务执行体把失败原因写进结果表（error），成功时写产物信息（md_path 等）
    error_message = get_task_result(task_id, "error", "")
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": status,  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表（中文展示名）
        "running_list": get_running_task_list(task_id),  # 正在运行的节点/阶段列表（中文展示名）
        "error": error_message,  # 失败原因（ImportProcessError.to_dict 的 message）
        "result": {
            "md_path": get_task_result(task_id, "md_path", ""),
            "file_title": get_task_result(task_id, "file_title", ""),
            "item_name": get_task_result(task_id, "item_name", ""),
            "chunk_count": get_task_result(task_id, "chunk_count", ""),
        },
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{status}，已完成节点：{task_status_info['done_list']}"
    )
    return task_status_info


# 7. 产物浏览接口：在浏览器里直接打开/下载导入生成的 Markdown
# 访问地址：http://localhost:8000/result/{task_id}/md   （?download=1 触发下载）
# 说明：md 正文里的图片引用已被 node_md_img 替换为 MinIO 直链
#       （http://localhost:9100/knowledge-base/upload-images/<文档名>/xxx.png，桶已开匿名只读），
#       所以这里只需把 md 本身吐出去，图片由浏览器直接向 MinIO 取，本服务不做图片代理。
@app.get("/result/{task_id}/md", summary="查看导入生成的Markdown")
async def get_import_markdown(task_id: str, download: int = 0):
    """
    :param task_id: 任务ID
    :param download: 传 1 时作为附件下载，否则浏览器内联展示
    """
    md_path = get_task_result(task_id, "md_path", "")
    if not md_path:
        raise HTTPException(
            status_code=404,
            detail=f"任务 {task_id} 尚无 Markdown 产物（请等待导入完成，或查看 /status 的 error 字段）",
        )

    resolved = Path(md_path).resolve()
    # 路径校验：产物必须落在 DATA_BASED_ROOT_DIR 内，防止结果表被写入任意路径后当文件读取器用
    root = Path(_data_root_dir()).resolve()
    if root not in resolved.parents and resolved != root:
        raise HTTPException(status_code=403, detail="产物路径不在数据根目录内，拒绝访问")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"产物文件不存在：{md_path}")

    return FileResponse(
        path=str(resolved),
        media_type="text/markdown; charset=utf-8",
        filename=resolved.name if download else None,
    )


# 6. 登录鉴权接入（实现在 web/api/auth_routes.py，此处只按契约挂载）：
#    - 未登录访问 page_routes 里的页面 → 302 /login.html?next=...
#    - 未登录调用 api_prefixes 前缀的接口 → 401 {"ok":false,"error":"未登录或会话已过期"}，
#      请求在中间件层被挡掉，业务逻辑（/upload 的落盘与 MinIO、后台导入图、/status 查询）
#      根本不会执行。
#    - 放行 /login.html /login /logout /api/me /health /docs /openapi.json /redoc。
#    页面绝对路径用 __file__ 锚定到项目根（不依赖启动时的 cwd）。
#    CORS 现状（具体源白名单 + allow_origin_regex + allow_credentials=True）本就允许
#    cookie 跨源携带，与登录会话兼容；页面默认同源（import.html 的 API_BASE 为空串走
#    相对路径），无需改动。install_auth 之后，上方各业务路由函数体一律未改动。
_PAGE_DIR = Path(__file__).resolve().parents[2] / "web" / "page"
install_auth(
    app,
    page_routes={
        "/import.html": _PAGE_DIR / "import.html",
        "/": _PAGE_DIR / "import.html",
    },
    api_prefixes=("/upload", "/status", "/result"),
)


if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("BOSS智库服务启动中...")
    # 注意：本项目必须从项目根用 -m 方式启动（`python -m uvicorn web.api.import_service:app --port 8000`），
    # 直接运行本文件可能因 sys.path 不含项目根而报 No module named 'config'。
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    # 端口分配：导入 8000 / 检索 8001（见 web/api/query_service.py 的 __main__），两者不冲突。
    # 注意：任务状态表在进程内存（utils/task_utils），所以这里不能开 --reload 以外的多 worker，
    # 否则 /upload 与 /status 落到不同进程会查不到任务。
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8000  # 服务端口
    )
