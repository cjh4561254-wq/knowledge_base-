# utils/sse_utils.py
"""
SSE（Server-Sent Events）会话通道管理 + FastAPI StreamingResponse 生成器。

本文件解决的核心问题：**队列生命周期竞态**（上一版实现的两处实测缺陷）
1. 覆盖式建队列 + 无条件删队列：
   旧实现 `create_sse_queue` 每次 `queue.Queue()` 直接覆盖注册表，而
   `sse_generator` 的 `finally` 无条件 `remove_sse_queue(session_id)`。
   前端在同一个 session 上连问两轮时（chat.html 每次 onSend 都 new 一个
   EventSource），第 1 轮的生成器往往还没退出（它靠 is_disconnected 或下一次
   轮询才发现连接已断），第 2 轮已经建了新队列；第 1 轮生成器退出时会把
   **第 2 轮的队列删掉**，此后所有 push_to_session 都找不到队列 → 日志出现
   `SSE push dropped` → 本轮 progress/delta/final 全丢，前端挂在「流式中转圈」。
2. 一个 session 只有一个「单消费」queue.Queue：
   两个浏览器标签同时连同一个 session 时，一条帧只会被其中一个标签读走
   （queue.Queue 是工作窃取语义），另一个标签永远等不到 final；而且先退出的
   那个标签会顺手把注册表删了，另一个标签后续事件全哑 —— 相当于「一个标签的
   退出关掉了另一个标签的流」。

现在的做法（对外函数签名、SSE 帧格式一律不变）：
- 每个 session 注册一个 **广播通道** `_SseQueue`（继承 queue.Queue 以保持既有
  类型契约与外部 `.put()` 用法）：`push_to_session` 把帧**复制**给该轮每一个
  订阅者（每个 HTTP 连接 = 一个订阅者 = 一条私有队列），互不窃取。
- 清理按**身份**（订阅者对象 + 通道对象双重比对）而不是按 session_id：
  生成器退出只摘掉自己那份订阅；整条通道只在「本轮已收到 CLOSE 且再没有任何
  订阅者」时才注销，且注销前确认注册表里存的仍是自己这个对象。
  因此旧生成器退出永远不可能误删新一轮仍在用的通道。
- `create_sse_queue` 语义明确为 **复用（幂等）**：同 id 重复调用返回同一通道，
  只把「轮次」向前推进一格（见函数 docstring 的取舍理由）。
- 生产者侧**永不阻塞**：扇出走 `put_nowait`，订阅者队列满了就丢最旧一帧
  （保住流尾的 final/CLOSE），检索线程不可能被慢客户端/断开的客户端卡住。

生成器的四个终止条件（都不留悬挂连接，每条都以 done 帧收口）：
1. 收到本轮 `__close__` → 发 `done{completed}`（前端 chat.html 的兜底终止帧）；
2. 客户端断开（`request.is_disconnected()`）→ 直接退出，不再写帧；
3. 本轮被新一轮取代（同 session 又发了一次 /query）→ 发 `done{superseded}`；
4. 静默超过 `_MAX_NO_EVENT_SECONDS` / 总时长超过 `_MAX_STREAM_SECONDS`
   → 分别发 `done{idle_timeout}` / `done{timeout}`，防止生产者中途死掉导致挂流。
回收：CLOSE 到达时若还没有任何消费者，通道**不当场注销**（否则「图跑得比浏览器
建连还快」的短答案分支会把 final 一起丢掉），改由「最后一个订阅者退订」或
「建轮/建连时的 TTL 清扫 `_gc_stale_locked`」回收，废弃会话驻留被 TTL 封顶。
"""
import asyncio
import json
import queue
import threading
import time
from collections import deque
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional, Tuple

from fastapi import Request

from tool.logger import logger


class SSEEvent:
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    DONE = "done"           # 流结束信号（服务端收到 CLOSE 后补发，给客户端明确的终止帧）
    CLOSE = "__close__"     # 关闭连接信号（内部事件，不会作为业务事件发给前端）


# ---------------------------------------------------------------------------
# 阈值（全部是纯内存/时间常量，不依赖配置文件，主控如需外指可统一改成 env）
# ---------------------------------------------------------------------------
# 空闲心跳阈值（秒）：连续 N 秒队列无消息就发一个 SSE 注释帧保活
# 为什么：图执行到长耗时节点（重排/向量检索）期间队列可能长时间无输出，
# 没有心跳时反向代理/浏览器可能判定连接空闲而掐断 SSE
_HEARTBEAT_IDLE_SECONDS = 15

# 无事件挂流上限（秒）：连续这么多秒**一个业务事件都没收到**就主动收流。
# 为什么必须有：CLOSE 是生产者兜底发的（query_service.run_query_graph 的 finally），
# 但若生产线程本身被杀/死锁（例如后台任务异常退出前进程被重启），没有这个上限
# 生成器会永远挂在轮询上，成为一个既不产出也永不释放的悬挂连接。
# 180s 远大于本机实测最慢的单节点耗时（CPU 重排 + Milvus 检索都是秒级），
# 期间任一事件都会把计数清零，所以正常问答不会被误杀。
_MAX_NO_EVENT_SECONDS = 180

# 单条 SSE 流的绝对寿命上限（秒）：兜底中的兜底，防止「一直有事件但永不 CLOSE」。
_MAX_STREAM_SECONDS = 900

# 无消费者期间的帧缓冲上限：/query 先建队列并启动后台图任务、EventSource 之后才连上，
# 这中间产生的 progress 必须缓存，否则本轮首帧进度会丢。有界是为了防「无人订阅
# 的会话把 delta 一直堆到 OOM」。
_BACKLOG_MAX = 512

# 单个订阅者的在途帧上限：满了丢最旧一帧（而不是丢最新，因为流尾的 final/CLOSE
# 必须送达；也不是阻塞等待，因为绝不能让检索线程为客户端的网速/卡顿买单）。
_SUBSCRIBER_MAXSIZE = 1024

# 丢弃告警节流：客户端卡死时每一帧都会触发一次「丢最旧」，若照单全发日志，
# 生产者线程就要为成千上万条 log 做文件 I/O —— 实测能把单次 push 从 0.05ms
# 拉到 26ms，正好违背「检索线程绝不能被 SSE 拖住」。所以只在第 1 次和每
# _DROP_LOG_EVERY 次丢弃时告警，连接结束时再补一条汇总。
_DROP_LOG_EVERY = 500

# 「本轮已收到 CLOSE 但始终没有消费者连上来」的通道保留时长（秒）。
# 为什么不能收到 CLOSE 就当场注销：/query 先建队列并启动后台图任务，EventSource 是
# 拿到 POST 响应之后才建的。短答案/拒答分支能在浏览器建连之前就把 final+CLOSE 推完，
# 此时队列里那批帧正是这个标签唯一能看到的答案 —— 当场删掉就等于把答案吞了
# （旧实现因为队列天然留着，反而没这个问题）。所以改成「保留到 TTL 到点，由下一次
# 建轮/建连顺带清扫」，既不吞答案，又把废弃会话的内存占用限制在 5 分钟内。
_CLOSED_CHANNEL_TTL_SECONDS = 300

# 「既没有消费者、又长时间没有任何活动」的通道保留时长（秒）：兜底回收生产线程
# 中途死掉（图没跑到 CLOSE）的僵死会话，防止注册表无限增长。正常一轮里每个节点
# 打点都会刷新活动时间，900s 静默且无人连接的会话只可能是死的。
_IDLE_CHANNEL_TTL_SECONDS = 900

# 轮询节拍（秒）：sync 队列 -> async 生成器的桥接等待时长。
# 1s 既能让生成器在客户端断开后尽快退出（每轮询一次检查一次 is_disconnected），
# 又给心跳/挂流计数提供了「1 tick = 1 秒」的简单节拍。
_POLL_TIMEOUT_SECONDS = 1.0

# 全局互斥锁：保护注册表 _session_stream 以及每条通道的订阅者表/轮次状态。
# 用一把全局 RLock（与 utils/task_utils.py 同风格）而不是「注册表锁 + 每通道锁」，
# 是为了从根上排除双锁的 AB-BA 死锁：本模块所有临界区都只是纯内存的
# append/pop/遍历（微秒级），任何 blocking 等待（queue.get(timeout=...)）都在锁外，
# 单进程几个并发会话的规模下不存在锁竞争问题。
_lock = threading.RLock()

# 通道序号：给每个通道一个单调递增的 generation token，用于「按身份删除」的日志与
# 跨进程排查（注册表身份比对用的是对象本身 is，见 _retire_locked）。
_channel_seq = 0

# 全局 SSE 会话通道注册表
# Key: session_id, Value: _SseQueue（queue.Queue 子类，故既有类型注解仍然成立）
_session_stream: Dict[str, "_SseQueue"] = {}


class _Subscriber:
    """
    一个 SSE 连接（一个标签页 / 一次 EventSource）对应一份订阅。

    round 字段是「这个连接属于第几轮」的快照：
    只有与通道当前轮次相同的订阅者才会收到新帧，于是上一轮遗留的连接不会
    把本轮的 delta 渲染进旧气泡，也不会被上一轮的迟到 CLOSE 提前掐断。
    """

    __slots__ = ("qid", "round", "q", "dropped")

    def __init__(self, qid: int, round_id: int) -> None:
        self.qid = qid
        self.round = round_id
        self.q: "queue.Queue" = queue.Queue(maxsize=_SUBSCRIBER_MAXSIZE)
        self.dropped = 0  # 因该连接消费不过来被丢弃的帧数（只用于告警）


class _SseQueue(queue.Queue):
    """
    session 级 SSE **广播通道**（一 session 一条，多个连接共享）。

    为什么继承 queue.Queue：
    - 对外契约 `get_sse_queue(...) -> Optional["queue.Queue"]` / `create_sse_queue(...)
      -> "queue.Queue"` 的类型注解必须继续成立（web/api/import_service.py 会拿它做
      `is not None` 存在性判断，别的包也可能直接 `.put(...)` 推帧）；
    - 继承后 `.put()/.put_nowait()` 就是「推事件」的既有写法，任何外部调用点不改一行
      代码即可获得扇出语义；
    - 父类那份 deque 缓冲**刻意不使用**：为了兼容「直接读通道」的旧用法（测试脚本 /
      将来的排查脚本 `get_sse_queue(sid).get_nowait()`），`get/get_nowait/qsize/empty`
      被改写为读写「本轮无消费者期间的缓冲帧」（self.backlog），语义与旧单队列
      在「单消费者」下的可观察行为一致。

    轮次（round）的作用：
    前端在同一会话里连续提问复用同一个 session_id，因此通道是长命的、轮次才是本轮的。
    `create_sse_queue` 每被调一次 = 开一轮（round += 1、清空上一轮缓冲、CLOSE 状态复位）。
    """

    def __init__(self, session_id: str) -> None:
        # 调父类构造只为把继承关系做实；父类的 deque 永远不会被本类写入。
        super().__init__()
        self.session_id = session_id
        self.token = 0                      # generation token（单调递增，仅用于日志/排查）
        self.round = 0                      # 当前轮次（>=1 才是活跃轮）
        self.subscribers: List[_Subscriber] = []
        # (round, item) 二元组：本轮还没有任何消费者时的缓冲，供第一个连上的消费者回放
        self.backlog: Deque[Tuple[int, Dict[str, Any]]] = deque(maxlen=_BACKLOG_MAX)
        self.closed = True                  # 本轮是否已收到 CLOSE（决定通道能否退休）；初始视为已结束
        self.closed_at: Optional[float] = None  # 本轮收到 CLOSE 的时刻（TTL 清扫判据）
        self.last_activity = time.monotonic()   # 最近一次建轮/推帧/订阅时刻（僵死轮次的清扫判据）
        self.sub_seq = 0                    # 订阅者编号发生器
        self.dropped_total = 0              # 本通道累计丢弃帧数（节流告警用）

    # ---------------- 轮次管理 ----------------
    def start_round_locked(self) -> int:
        """开新一轮：轮次 +1、复位 CLOSE 标记、丢掉上一轮的无人消费缓冲。"""
        self.round += 1
        self.closed = False
        self.closed_at = None
        self.last_activity = time.monotonic()
        self.backlog.clear()
        return self.round

    # ---------------- 生产者侧（扇出，永不阻塞） ----------------
    def put(self, item, block: bool = True, timeout: Optional[float] = None) -> None:
        """
        兼容 Queue.put 签名，但 **block/timeout 一律忽略**：广播通道的投递按定义
        不排队、不等待、不阻塞，生产者（检索线程）永远不会被 SSE 拖住。
        """
        self.put_nowait(item)

    def put_nowait(self, item) -> None:  # type: ignore[override]
        dropped_pairs: List[Tuple[int, int]] = []  # (订阅者 qid, 该订阅者累计丢弃数)
        should_log = False
        event_name = None
        if isinstance(item, dict):
            event_name = item.get("event")
        with _lock:
            self.last_activity = time.monotonic()
            round_id = self.round
            targets = [s for s in self.subscribers if s.round == round_id]
            for sub in targets:
                if not self._offer_locked(sub, item):
                    dropped_pairs.append((sub.qid, sub.dropped))
            if dropped_pairs:
                self.dropped_total += len(dropped_pairs)
                # 只在首次和每 _DROP_LOG_EVERY 次越界时告警（节流，见常量注释）
                bucket = self.dropped_total // _DROP_LOG_EVERY
                logged_bucket = (self.dropped_total - len(dropped_pairs)) // _DROP_LOG_EVERY
                should_log = bucket != logged_bucket or self.dropped_total == len(dropped_pairs)
            # 本轮还没有消费者时留一份缓冲，等 EventSource 连上来回放（否则
            # /query 到 stream 之间的 progress 会永久丢失）
            if not targets:
                self.backlog.append((round_id, item))
            if event_name == SSEEvent.CLOSE:
                # 本轮到此为止。**注意：这里绝不当场注销通道** ——
                # 「图跑得比浏览器建连还快」（拒答/短答案分支 + EventSource 还在排队）时，
                # CLOSE 到达时一个订阅者都没有，若此刻 retire，缓冲里的 final 就被一起丢了，
                # 随后连上的标签只能拿到 session_not_found，等于把答案整条吞掉（旧实现不会，
                # 因为旧队列本来就留着）。通道的退休改由「消费者退订时」+「TTL 清扫」决定。
                self.closed = True
                self.closed_at = self.last_activity
        # 告警放到锁外：logger 带文件 handler，不做无谓的锁内 I/O
        if should_log:
            for qid, cnt in dropped_pairs:
                logger.warning(
                    f"SSE subscriber queue full, dropped oldest: session={self.session_id} "
                    f"sub={qid} sub_dropped={cnt} channel_dropped={self.dropped_total}"
                )

    @staticmethod
    def _offer_locked(sub: _Subscriber, item) -> bool:
        """
        把一帧放进订阅者私有队列。返回 False 表示发生了「丢最旧」。
        满队列时丢最旧而不是丢最新：流尾的 final / CLOSE 必须送达，
        否则客户端收不到终止帧会永久挂流（比少显示一帧旧 delta 严重得多）。
        """
        try:
            sub.q.put_nowait(item)
            return True
        except queue.Full:
            pass
        try:
            sub.q.get_nowait()
        except queue.Empty:
            pass
        sub.dropped += 1
        try:
            sub.q.put_nowait(item)
        except queue.Full:
            pass
        return False

    # ---------------- 消费者侧（订阅/退订，全部按身份） ----------------
    def subscribe_locked(self) -> _Subscriber:
        """
        登记一个连接为「本轮订阅者」，并把本轮**已缓冲**的帧交给它。

        缓冲只存在于「本轮还没有任何消费者」的窗口（见 put_nowait），所以回放
        不会和实时投递重复；这里把取走的条目从缓冲里摘掉，保证同一轮第二个
        连接不会把同一批帧再收一遍。
        """
        self.sub_seq += 1
        sub = _Subscriber(self.sub_seq, self.round)
        self.subscribers.append(sub)
        self.last_activity = time.monotonic()
        if self.backlog:
            keep: List[Tuple[int, Dict[str, Any]]] = []
            for buffered_round, buffered_item in self.backlog:
                if buffered_round == sub.round:
                    self._offer_locked(sub, buffered_item)
                else:
                    keep.append((buffered_round, buffered_item))
            self.backlog = deque(keep, maxlen=_BACKLOG_MAX)
        return sub

    def unsubscribe_locked(self, sub: _Subscriber) -> bool:
        """
        只摘掉**自己那一份**订阅（按对象身份，绝不按 session_id 顺手删全局）。
        返回 True 表示「整条通道此刻可以退休」。
        """
        try:
            self.subscribers.remove(sub)
        except ValueError:
            # 重复退订 / 已被别的逻辑清理：幂等，忽略
            pass
        return self.closed and not self.subscribers

    def take_backlog_item_locked(self) -> Dict[str, Any]:
        """从本轮缓冲里取最早一帧；取不到抛 queue.Empty（兼容旧单队列的读法）。"""
        for i, (round_id, item) in enumerate(self.backlog):
            if round_id == self.round:
                del self.backlog[i]
                return item
        raise queue.Empty

    # ---------------- 兼容旧用法的读接口（项目内生产代码不再使用） ----------------
    def get_nowait(self):  # type: ignore[override]
        with _lock:
            return self.take_backlog_item_locked()

    def get(self, block: bool = True, timeout: Optional[float] = None):  # type: ignore[override]
        """
        兼容 `get_sse_queue(sid).get(timeout=1)` 这类排查/测试写法。
        真正的 SSE 连接走 subscribe_locked 的私有队列，不会读到这里。
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                return self.get_nowait()
            except queue.Empty:
                if not block:
                    raise
                if deadline is not None and time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)

    def qsize(self) -> int:  # type: ignore[override]
        with _lock:
            return len([1 for round_id, _ in self.backlog if round_id == self.round])

    def empty(self) -> bool:  # type: ignore[override]
        return self.qsize() == 0


def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定 session 的队列（实为广播通道，仍是 queue.Queue 实例）"""
    with _lock:
        return _session_stream.get(session_id)


def create_sse_queue(session_id: str) -> "queue.Queue":
    """
    创建（或复用）指定 session 的 SSE 队列，并**开启新一轮**。

    语义选择：**复用（幂等）**，而不是「覆盖式新建」。理由三条：
    1. EventSource 是 session 级连接：前端在同一会话连问第二轮时，上一轮的连接/
       生成器可能还没退出，重连中的浏览器也可能正挂在这个 session 上。若这里换掉
       队列对象，「生产者写入新队列、旧消费者读旧队列」的分裂就重新出现了 ——
       这正是本次要修的竞态的镜像面。
    2. 与 utils/task_utils.reset_task 的取舍保持一致：那个函数明确「只清三本账、
       不销毁也不排空队列」，因为它销毁队列会让后续 push_to_session 找不到队列而
       静默丢事件。队列层如果每轮重建，就把它极力避免的事情又做了一遍。
    3. 「本轮从哪开始」不该由队列对象身份决定，而由 **轮次(round) + __close__/done
       帧** 决定：本函数把 round 前进一格、清掉上一轮的无人消费缓冲、复位 CLOSE 标记；
       每个连接在订阅时记下自己属于哪一轮，旧轮的迟到帧不会再污染新连接。
       于是「旧生成器退出」只是摘掉自己那份订阅，不可能删掉新一轮正在用的通道。

    返回值仍是 queue.Queue 实例（本模块的 _SseQueue），调用方无需感知实现。
    """
    global _channel_seq
    with _lock:
        _gc_stale_locked()   # 建轮是天然的回收点：顺带清掉 TTL 到期的废弃会话
        channel = _session_stream.get(session_id)
        if channel is None:
            _channel_seq += 1
            channel = _SseQueue(session_id)
            channel.token = _channel_seq
            _session_stream[session_id] = channel
        channel.start_round_locked()
        logger.debug(
            f"SSE channel ready: session={session_id} token={channel.token} round={channel.round}"
        )
        return channel


def remove_sse_queue(session_id: str):
    """
    按 session_id **强制**注销队列（保留既有对外 API）。

    注意：`sse_generator` 的 finally **不再调用**本函数 —— 它是按 id 无条件删除的，
    正是旧版误删新一轮队列的元凶。生成器改用 `_unsubscribe`（按订阅者身份摘除）+
    `_retire_locked`（按通道对象身份注销）。本函数留给「显式结束会话/测试清理」用。
    """
    with _lock:
        _session_stream.pop(session_id, None)


def _retire_locked(session_id: str, channel: _SseQueue) -> bool:
    """
    身份安全的注销：只有注册表里存的**仍是这个通道对象**时才删。

    若期间已有别的通道被注册到同一个 session_id（强制 remove 后又新建、
    或将来把 create 语义改成替换），这里会静默跳过，绝不删掉别人的东西。
    """
    with _lock:
        if _session_stream.get(session_id) is channel:
            _session_stream.pop(session_id, None)
            logger.debug(f"SSE channel retired: session={session_id} token={channel.token}")
            return True
        return False


def _gc_stale_locked() -> int:
    """
    清扫「已经没有订阅者、且本轮已结束太久 / 久无活动」的通道（调用方需持锁）。

    为什么挂在建轮 / 建连这两个点上（而不是后台定时线程）：
    - 本模块是单进程内存态，会话数量小，不需要常驻线程和额外的生命周期；
    - 通道的产生只来自 /query（建轮）与 /stream（建连），这两个点就是天然的回收时机，
      再配上 _IDLE/_CLOSED 两个 TTL，废弃会话的驻留时间被硬性封顶，不会累积。
    返回清扫掉的数量。
    """
    now = time.monotonic()
    stale: List[str] = []
    for sid, channel in _session_stream.items():
        if channel.subscribers:
            continue  # 还有连接挂着，绝不回收
        if channel.closed and channel.closed_at is not None \
                and (now - channel.closed_at) > _CLOSED_CHANNEL_TTL_SECONDS:
            stale.append(sid)
        elif (now - channel.last_activity) > _IDLE_CHANNEL_TTL_SECONDS:
            stale.append(sid)
    for sid in stale:
        channel = _session_stream.get(sid)
        if channel is not None and not channel.subscribers:
            _session_stream.pop(sid, None)
            logger.debug(
                f"SSE channel garbage-collected: session={sid} token={channel.token} "
                f"round={channel.round} closed={channel.closed}"
            )
    return len(stale)


def _subscribe(session_id: str) -> Optional[Tuple[_SseQueue, _Subscriber]]:
    """内部：为一次 HTTP 连接登记订阅。返回 (通道, 订阅者)；会话不存在返回 None。"""
    with _lock:
        _gc_stale_locked()
        channel = _session_stream.get(session_id)
        if channel is None or channel.round <= 0:
            # round<=0 理论上不会出现（create 即开轮），留作防御
            return None
        return channel, channel.subscribe_locked()


def _unsubscribe(channel: _SseQueue, sub: _Subscriber) -> None:
    """内部：生成器退出时清理 —— 只摘自己那份订阅；整条通道仅在
    「本轮已 CLOSE 且再无订阅者」且「注册表里仍是本通道」时才注销。"""
    with _lock:
        should_retire = channel.unsubscribe_locked(sub)
        # 把该订阅者残留的帧丢掉，释放引用（旧轮次被取代的连接尤其需要这一步）
        while True:
            try:
                sub.q.get_nowait()
            except queue.Empty:
                break
        if should_retire:
            _retire_locked(channel.session_id, channel)
    if sub.dropped:
        # 汇总告警（连接级一次性，生产者路径上已经节流）
        logger.warning(
            f"SSE subscriber detached: session={channel.session_id} sub={sub.qid} "
            f"round={sub.round} dropped={sub.dropped} (客户端消费过慢，已丢最旧帧)"
        )


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """
    打包 SSE 消息格式：`event: <事件名>\\ndata: <JSON>\\n\\n`

    为什么这样写：
    - 前端 chat.html 用 EventSource 的 addEventListener('progress'|'delta'|'final'|'error')
      按【命名事件】分发，所以必须带 `event:` 行，只发 data 行会全部落到 onmessage 而无人响应。
    - ensure_ascii=False 保证中文以 UTF-8 原样输出（EventSource 规范固定按 UTF-8 解码，
      json.dumps 默认 ensure_ascii=True 会把中文变成 \\uXXXX，体积膨胀且日志不可读）。
    - default=str 兜底：节点推送的 data 里若混入 ObjectId/datetime 等非可序列化对象，
      在生成器 yield 时抛 TypeError 会直接炸掉整条 SSE 流；退化成字符串保证流不中断。
    - 换行安全：json.dumps 会把字符串内的 \\n 转义，data 负载恒为单行，符合 SSE 帧协议。
    """
    payload = json.dumps(data if data is not None else {}, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """
    通过 session_id 推送事件（生产者入口，被 LangGraph 并行节点与后台线程调用）。

    线程安全 + 永不阻塞：扇出走 _SseQueue.put_nowait（内部 put_nowait + 满了丢最旧），
    不存在「客户端断开/消费慢 → 检索线程被队列卡死」的可能；本函数内部的任何异常
    也都在这里吞掉，SSE 永远不能把检索流程带崩。
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        # 记录丢事件：多半是客户端已断开/会话队列已被清理，留痕便于排查"前端收不到某事件"
        logger.warning(f"SSE push dropped (no queue for session): session={session_id} event={event}")
        return
    try:
        stream_queue.put({"event": event, "data": data})
    except Exception as e:  # noqa: BLE001 - SSE 故障绝不能影响主流程
        logger.warning(f"SSE push failed: session={session_id} event={event} err={e}")


async def sse_generator(session_id: str, request: Request) -> AsyncGenerator[str, None]:
    """
    SSE 生成器，用于 FastAPI 的 StreamingResponse

    生命周期（本模块竞态修复的关键）：
    - 进入：`_subscribe` 登记「本轮的一个订阅者」，并取回本轮无人消费期间的缓冲帧；
    - 运行：只从**自己那份私有队列**读帧（多个标签各读各的，互不窃取、互不影响）；
      帧在扇出时就按 round 过滤过 —— 只有本轮的订阅者才收得到本轮的帧；
      发现轮次已被下一轮取代就补发 done 收流；
    - 退出：`finally` 里 `_unsubscribe` **按对象身份**摘掉自己那份订阅，
      只有「本轮已收到 CLOSE 且已无任何订阅者且注册表里仍是本通道」时才注销通道。
      绝不再无条件 `remove_sse_queue(session_id)`。
    """
    grabbed = _subscribe(session_id)
    if grabbed is None:
        # 队列不存在（会话已结束被清理 / 客户端重连过快照已失效）。
        # 为什么要多发这两帧：EventSource 在服务端主动断流后会自动重连，
        # 若这里立刻 return（0 字节响应），浏览器会按默认 ~3 秒间隔无限重连形成风暴；
        # 先下发超大 retry 间隔再补 done 终止帧，让"死会话"的连接安静下来。
        yield "retry: 3600000\n\n"
        yield _sse_pack(SSEEvent.DONE, {"status": "session_not_found"})
        return

    channel, sub = grabbed
    my_round = sub.round

    loop = asyncio.get_running_loop()
    idle_ticks = 0        # 距上次「发心跳」的秒数（发完心跳清零，所以它会循环归零）
    silent_ticks = 0      # 距上次「收到任何业务事件」的秒数（只有真事件才清零 -> 挂流上限判据）
    started = time.monotonic()
    try:
        # 发送连接建立信号
        yield _sse_pack(SSEEvent.READY, {})

        while True:
            # 若客户端断开，尽快退出
            if await request.is_disconnected():
                break

            # 绝对寿命兜底：无论有没有事件，超上限就明确收流，绝不留悬挂连接
            if time.monotonic() - started > _MAX_STREAM_SECONDS:
                yield _sse_pack(SSEEvent.DONE, {"status": "timeout"})
                break

            # 本轮已被下一轮取代（同 session 又发了一次 /query）：
            # 继续挂着只会把新轮的帧误当本轮渲染、或永远等不到本轮的 CLOSE，
            # 所以直接给客户端一个明确终止帧。
            with _lock:
                superseded = channel.round != my_round
            if superseded:
                logger.debug(
                    f"SSE stream superseded: session={session_id} round={my_round} "
                    f"current={channel.round}"
                )
                yield _sse_pack(SSEEvent.DONE, {"status": "superseded"})
                break

            try:
                # 使用 run_in_executor 避免阻塞 async 事件循环
                # 超时 1s：既能快速响应客户端断开，也为心跳计数提供节拍
                # 读的是 sub.q（本连接私有），不是通道本身 -> 多标签不会互相抢帧
                msg = await loop.run_in_executor(
                    None, sub.q.get, True, _POLL_TIMEOUT_SECONDS
                )
                silent_ticks = 0
                idle_ticks = 0
            except queue.Empty:
                idle_ticks += 1
                silent_ticks += 1
                # 心跳：长时间无业务事件时发送 SSE 注释帧（以 ':' 开头，EventSource 规范规定忽略）
                # 作用：让代理/防火墙认定连接活跃，且给服务端一次 is_disconnected 的检查机会
                if idle_ticks >= _HEARTBEAT_IDLE_SECONDS:
                    idle_ticks = 0
                    yield ": heartbeat\n\n"
                # 挂流保护：本轮生产者可不会再发 CLOSE（线程被杀等），到点主动收流
                if silent_ticks >= _MAX_NO_EVENT_SECONDS:
                    logger.warning(
                        f"SSE stream idle-closed (no event for {silent_ticks}s): session={session_id}"
                    )
                    yield _sse_pack(SSEEvent.DONE, {"status": "idle_timeout"})
                    break
                continue

            event = msg.get("event")
            data = msg.get("data")

            # 特殊关闭事件：先向客户端补发一个显式 done 终止帧再断流
            # 为什么：前端 chat.html 主要靠 final 关闭 EventSource；
            # 若流程异常导致 final 丢失、只有 __close__，客户端会挂在"还在流式"状态；
            # done 帧给了未来客户端一个统一的兜底终止信号（无监听者的命名事件会被浏览器忽略，向后兼容）
            if event == SSEEvent.CLOSE:
                yield _sse_pack(SSEEvent.DONE, {"status": "completed"})
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        # 生成器被取消/对端断开：静默退出
        return
    finally:
        # 清理资源：按「订阅者身份」摘除自己那一份，而不是按 session_id 删整条队列。
        # 这一步是本轮竞态修复的落点：上一轮生成器无论何时退出，都不可能删掉
        # 新一轮正在使用的通道。
        _unsubscribe(channel, sub)
