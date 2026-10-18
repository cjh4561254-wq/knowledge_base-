# utils/task_utils.py
"""
内存态任务追踪 + LangGraph 节点打点（running/done）。

为什么需要锁：这些函数会被 LangGraph 多线程节点并发调用（导入流程后台线程、
检索流程的并行节点 node_search_embedding / node_search_embedding_hyde /
node_web_search_mcp 等同时操作同一个 task_id 的 dict/list）。list/dict 的
"读-改-写"组合（如 `if node_name not in running: running.append(...)`）不是
原子操作，无锁时会出现丢点、重复点甚至遍历时字典变形的竞态。
用可重入锁 RLock：task_push_queue 内部还会再调 get_*_list（同线程重入）。
"""
import threading
from typing import Dict, List

from .sse_utils import push_to_session

# 全局互斥锁：保护下面 4 个共享 dict 的一切读写
_lock = threading.RLock()

# ---------------------------
# 内存态任务追踪（单进程）
# ---------------------------
# key: task_id
# value: 节点名列表（原始英文/节点ID）
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}

# key: task_id
# value: status 字符串（如 pending/processing/completed/failed）
_tasks_status: Dict[str, str] = {}

# key: task_id
# value: 任务结果（例如 query 的 answer）
_tasks_result: Dict[str, Dict[str, str]] = {}


TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点名 -> 中文名映射（用于前端展示）
# 说明：这里的 key 应与 LangGraph 的 add_node("xxx", ...) 中的节点名一致。
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_kg": "导入知识图谱",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    # --- Query 流程节点（kb/query_process/main_graph.py）---
    "node_item_name_confirm": "确认问题产品",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_query_kg": "查询知识图谱",
    "node_join": "多路搜索合并",
}


def _ensure_task(task_id: str) -> None:
    """确保 task_id 对应的数据结构已初始化。（调用方需持有 _lock）"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_cn(node_name: str) -> str:
    """将节点名转换为中文展示名；若无映射则返回原名。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    添加“正在运行”的节点任务。

    参数：
    - task_id: 任务ID（导入流程为 task_id，检索流程为 session_id）
    - node_name: 节点名称(节点ID)
    - is_stream: 检索流式场景为 True，立刻向 SSE 队列推送进度
    """
    with _lock:
        _ensure_task(task_id)
        running = _tasks_running_list[task_id]
        # 避免重复追加
        if node_name not in running:
            running.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    添加“已完成”的节点任务。

    注意：添加已完成任务时，会把同名的“正在运行”任务删除。

    参数：
    - task_id: 任务ID（调用task_id/session_id 标记任务）
    - node_name: 节点名称(节点ID)
    - is_stream: 检索流式场景为 True，done 事件通过 task_push_queue 推给
      SSE 消费者（前端进度条）；流式的 final/close 兜底由 node_answer_output
      直接 push_to_session 完成，这里只负责 progress 事件。
    """
    with _lock:
        _ensure_task(task_id)

        # 1) 从 running 中移除同名节点（可能出现重复，移除所有）
        running = _tasks_running_list[task_id]
        _tasks_running_list[task_id] = [n for n in running if n != node_name]

        # 2) 追加到 done（保持完成顺序），避免重复
        done = _tasks_done_list[task_id]
        if node_name not in done:
            done.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """
    存储任务结果字段（如 answer / error）。
    """
    with _lock:
        _ensure_task(task_id)
        _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """
    获取任务结果字段（如 answer / error）。
    """
    with _lock:
        _ensure_task(task_id)
        return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    """
    获取当前任务状态。

    参数：
    - task_id: 任务ID

    返回：
    - str: 状态名称；如果未设置过则返回空字符串
    """
    with _lock:
        return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """
    获取已完成节点列表（中文展示）。
    """
    with _lock:
        _ensure_task(task_id)
        done = _tasks_done_list.get(task_id, [])
        # 返回拷贝：避免调用方遍历时并发节点写入同一 list
        return [_to_cn(n) for n in done]


def get_running_task_list(task_id: str) -> List[str]:
    """
    获取正在运行节点列表（中文展示）。
    """
    with _lock:
        _ensure_task(task_id)
        running = _tasks_running_list.get(task_id, [])
        # 返回拷贝：避免调用方遍历时并发节点写入同一 list
        return [_to_cn(n) for n in running]


def update_task_status(
    task_id: str, status_name: str, push_queue: bool = False
) -> None:
    """
    更新任务状态。

    参数：
    - task_id: 任务ID
    - status_name: 状态名称（字符串，取值见 TASK_STATUS_*）
    - push_queue: 是否立刻向 SSE 队列推送一次 progress
    """
    with _lock:
        _tasks_status[task_id] = status_name

    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    """
    向 SSE 消费者推送 progress 事件（status + done/running 列表快照）。

    为什么先加锁取快照：done/running 正被并行节点并发写，不加锁可能推出
    半更新（同一节点既 running 又 done）的进度，前端进度条会闪跳。
    """
    with _lock:
        payload = {
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        }
    # 队列入队本身线程安全（queue.Queue），放到锁外，缩短临界区
    push_to_session(task_id, "progress", payload)


def clear_task(task_id: str):
    """清理任务的全部内存态数据（前端刷新 / 任务结束后调用，防内存泄漏）。"""
    with _lock:
        _tasks_running_list.pop(task_id, None)
        _tasks_done_list.pop(task_id, None)
        _tasks_status.pop(task_id, None)
        _tasks_result.pop(task_id, None)


def reset_task(task_id: str, reset_status: bool = True) -> None:
    """
    把「某个会话的本轮进度账本」清零，供每一轮图执行开始之前调用。

    为什么必须有这个函数（实测 bug）：
        检索侧的账本 key 是 session_id，而前端在同一个会话里连续提问复用同一个
        session_id。add_running_task / add_done_task 只做「追加 + 同名互斥」，
        从不整轮删除，于是第 2 问刚发起时 get_done_task_list 仍返回第 1 问的 7 个
        节点，SSE progress 帧直接是「已完成7，进行中0，状态：已完成」，节点清单里
        还挂着上一轮的 切片搜索/网络搜索/倒排融合/重排序。用户看到「全部已完成」，
        进度条失去意义，甚至会误以为本轮真的跑过检索。清掉三本账后本轮进度才从空开始。

    与 clear_task 的区别（语义不同，两者都保留，不可互相替换）：
        - clear_task：任务终态清理，把该 id 的 4 个 key 整体 pop 掉，用于「这个 id
          不再用了」；清完后 get_task_status 返回空串。
        - reset_task：新一轮开始前的重置，容器保留并置空，状态回落为 pending 而不是
          空串——task_push_queue 会把 status 一起发给前端，空串会被 chat.html 渲染成
          「状态：unknown」，而 pending 对应前端「等待中」，正是此刻的真实语义。

    参数：
    - task_id: 任务ID（导入流程为 task_id，检索流程为 session_id）
    - reset_status: 是否同时把状态回落为 TASK_STATUS_PENDING。
      默认 True（/query 里先清零、紧接着 update_task_status 置 processing）。
      在「同一轮内已经由 web 层置好 processing、只是想再兜底清一次账本」的位置传
      False，避免把已发出的 processing 语义回退成 pending。

    SSE 队列的取舍（关键，写清理由）：
        本函数【既不销毁也不排空】该 session 的 queue.Queue，只清三本账：
        1) 不销毁：EventSource 连的是 session 级队列，前端可能在上一轮之后仍然挂着
           这个连接（自动重连 / 多标签 / 尚未消费到 done 帧）。销毁队列会让后续
           push_to_session 找不到队列而静默丢事件，本轮进度反而全哑。
        2) 不排空：队列里若还压着上一轮未消费的帧，强行排空会连同那轮的 final /
           CLOSE 一起丢掉 —— 客户端收不到终止帧会永久挂在「流式中」并无间隔重连，
           比多显示一帧旧进度严重得多。残留旧帧的代价只是「前端先闪一下上一轮清单」，
           随后本轮任一节点打点或 update_task_status 推出的空清单帧就会覆盖它
           （chat.html 的 renderProgress 每次整体重写 done/running 两块，非增量追加）。

    线程安全：与本文件其它函数共用 _lock（这些函数会被 LangGraph 并行节点并发调用）。
    幂等：session_id 从未出现过时只建空容器、不抛异常；重复调用结果一致。
    """
    with _lock:
        # 建好空容器：本函数内部不调 task_push_queue，所以不能沿用 add_* 那套
        # 「只建缺失项」的懒初始化语义，必须显式把三个容器重置为空。
        _ensure_task(task_id)
        # 整体替换成新的空容器，而不是原地 .clear()：
        # 外部（task_push_queue 快照 / get_*_list 拷贝）拿到的都是旧 list 的副本，
        # 换引用不会让任何已发出的 payload 变成半成品，也不影响并发节点稍后重新追加。
        _tasks_running_list[task_id] = []
        _tasks_done_list[task_id] = []
        _tasks_result[task_id] = {}
        if reset_status:
            _tasks_status[task_id] = TASK_STATUS_PENDING
