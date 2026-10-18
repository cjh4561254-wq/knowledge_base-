# utils/reranker_http_utils.py
"""重排序(Rerank)统一入口：本地 Cross-Encoder 与在线 rerank API 的双模封装。

为什么需要这一层（而不是只保留原来的纯 HTTP 实现）：
1. 本机实测：SiliconFlow rerank 返回 402（无余额）、DashScope qwen3-rerank 返回 401
   （key 失效），在线接口在当前环境**完全不可用**。因此默认走本地
   bge-reranker-v2-m3（`reranker_config.rerank_mode=local` + `reranker_path`）。
   在线分支原样保留：日后换了可用 key，只改 .env 的 RERANK_MODE 即可，
   调用方（node_rerank）一行都不用动。
2. 惰性单例：本地权重约 2.27GB，绝对禁止在 import 期加载
   （本模块被 node_rerank 顶层 import，一旦顶层建模就等于所有检索节点启动即卡死）。
   因此 FlagEmbedding 与权重都在 `get_local_reranker()` 里首次调用时才加载，
   并用「双重检查 + 线程锁」保证 FastAPI/LangGraph 并发下只加载一次。
3. 量纲统一（关键）：断崖检测阈值 RERANK_GAP_ABS=0.5 / RERANK_GAP_RATIO=0.25
   是按 0~1 的相关性分设计的。在线 API 返回的就是 0~1 的 relevance_score，
   而 flagembedding 的 FlagReranker 默认返回**未归一化的 logits**（可达 ±几十），
   直接用会让"绝对差 ≥ 0.5"恒定成立、第 1 条之后立刻被截断。
   → 本地分支必须 `normalize=True`（内部做 sigmoid，输出 0~1）。
   注意：1.4.2 版本**没有** `compress_scores` 这个参数（那是 1.3 之前的老 API），
   对应能力在本版本里就叫 `normalize`，已用 inspect 现场核对签名。
4. 设备/精度：本机 torch 2.14.1+cpu、`cuda.is_available()=False`。
   构造参数名是 `devices`（不是 device）；CPU 上必须强制 `use_fp16=False`
   （库默认值是 True，半精度算子在 CPU 上直接报错）。代码里不出现任何 .to("cuda")。
5. 对外契约（node_rerank 依赖，勿改）：
   - `rerank_documents(query, documents) -> list[float]`：长度与 documents 严格一致、
     下标对齐（调用方用 zip 把分数贴回文档）。
   - 新增 `rerank(query, documents) -> list[dict]`：统一入口的"排序后文档"形态，
     每条 {"index": 原始下标, "document": 文本, "score": 相关性分}，按分数降序，
     与主流 rerank API 的 results 结构一致，便于其它调用方直接使用。
"""

import os
import threading
from typing import Any, Dict, List, Sequence

import dashscope
from dotenv import load_dotenv

from config.reranker_config import reranker_config
from tool.logger import logger

# override=True：系统环境变量里存在一个已失效的 OPENAI_API_KEY/BASE_URL，
# 不覆盖会污染 .env 的取值（与 config/*.py 的处理保持一致）
load_dotenv(override=True)

# ==================== 常量 ====================

# 在线模式标识：rerank_mode 取值 local / api（缺省 local，因为本机在线不可用）
MODE_LOCAL = "local"
MODE_API = "api"

# CPU 上单次打分的批大小：默认 16（候选文档通常 5~15 条，一批就能吃完；
# 沿用库默认 128 会在 CPU 上把显式内存/线程一次性打满，且没有收益）
_LOCAL_BATCH_SIZE = max(1, int(os.getenv("RERANKER_BATCH_SIZE", "16")))
# Cross-Encoder 的截断长度：bge-reranker-v2-m3 的位置编码上限就是 512 token，
# 传更大的值模型不会用（超出部分被截断），这里显式写出契约，避免"静默截断"看不懂
_LOCAL_MAX_LENGTH = max(64, int(os.getenv("RERANKER_MAX_LENGTH", "512")))

# ==================== 本地模型单例（惰性） ====================

# 模型单例对象：import 本模块不会赋值，第一次真正打分才加载权重
_local_reranker = None
# 初始化锁：并发检索时只允许一个线程真正加载权重
_reranker_lock = threading.Lock()
# 推理锁：CPU 上 torch 推理本身会吃满全部核，多个请求并发打分只会互相拖慢并放大内存占用，
# 因此串行化（在线模式不需要该锁，走网络 IO）
_infer_lock = threading.Lock()


def get_local_reranker():
    """
    获取本地 FlagReranker 单例（惰性加载 bge-reranker-v2-m3）。
    :return: 已完成权重载入、可直接 compute_score 的实例
    """
    global _local_reranker
    if _local_reranker is not None:
        return _local_reranker

    with _reranker_lock:
        # 双重检查：等锁期间可能已经有别的线程加载好了
        if _local_reranker is not None:
            return _local_reranker

        # 延迟 import：FlagEmbedding 会连带 import torch/transformers（秒级 + 数百 MB），
        # 放在模块顶层会让每个 import 本模块的节点都被拖慢
        from FlagEmbedding import FlagReranker

        model_path = (reranker_config.reranker_path or "").strip()
        if not model_path:
            raise RuntimeError("reranker_config.reranker_path 为空，无法使用本地 rerank（请检查 BGE_RERANKER_LARGE）")
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"本地 reranker 模型目录不存在：{model_path}")

        device = (reranker_config.reranker_device or "cpu").strip().lower()
        use_fp16 = bool(reranker_config.reranker_fp16)
        # 本机是 CPU 版 torch：fp16 相关算子不支持，配置被误改也必须强制关掉
        if "cuda" not in device:
            if use_fp16:
                logger.warning(f"reranker 设备为 {device}，不支持半精度，已强制 use_fp16=False")
            use_fp16 = False

        logger.info(
            f"首次加载本地 reranker：path={model_path} device={device} fp16={use_fp16} "
            f"batch_size={_LOCAL_BATCH_SIZE} max_length={_LOCAL_MAX_LENGTH}"
        )
        _local_reranker = FlagReranker(
            model_name_or_path=model_path,
            devices=device,  # 1.4.2 的参数名是 devices（不是 device）
            use_fp16=use_fp16,
            batch_size=_LOCAL_BATCH_SIZE,
            max_length=_LOCAL_MAX_LENGTH,
            query_max_length=_LOCAL_MAX_LENGTH // 4,  # 问题很短，留出预算给片段
            # sigmoid 归一到 0~1，与在线 API 同量纲，断崖检测阈值才有意义
            normalize=True,
        )
        return _local_reranker


def _resolve_mode() -> str:
    """
    解析当前 rerank 模式（每次调用都读配置，便于运行时改 .env / 单测里 monkeypatch）。
    :return: MODE_LOCAL 或 MODE_API
    """
    mode = (reranker_config.rerank_mode or MODE_LOCAL).strip().lower()
    # 兼容 .env 里可能写的 local_model / flag / bge 等写法：只要不是明确的 api，都走本地
    return MODE_API if mode in (MODE_API, "api", "http", "online", "dashscope") else MODE_LOCAL


def _normalize_documents(documents: Sequence[str]) -> List[str]:
    """
    把入参统一成 list[str]：None → 空串，非字符串 → str()。
    为什么必须做：下游是 zip(文档, 分数)，任何一条被丢弃都会导致整体错位；
    所以这里做的是「等长替换」而不是「过滤」。
    """
    if documents is None:
        return []
    return ["" if d is None else (d if isinstance(d, str) else str(d)) for d in documents]


# ==================== 分支实现 ====================

def _score_by_local(query: str, documents: List[str]) -> List[float]:
    """
    本地 Cross-Encoder 打分（FlagReranker.compute_score）。
    :return: 与 documents 等长、下标对齐的 0~1 相关性分
    """
    model = get_local_reranker()
    # compute_score 要求 [(query, passage), ...]；query 为 None 时兜底空串，
    # 否则 tokenizer 会因非字符串直接抛异常
    safe_query = query if isinstance(query, str) else ("" if query is None else str(query))
    sentence_pairs = [[safe_query, doc] for doc in documents]

    with _infer_lock:
        raw_scores = model.compute_score(sentence_pairs)

    scores = list(raw_scores or [])
    if len(scores) != len(documents):
        # 理论上不会发生（compute_score 与输入严格一一对应），真发生说明库行为变了
        logger.warning(f"本地 rerank 返回 {len(scores)} 分，与文档数 {len(documents)} 不一致，按位补齐")
        scores = (scores + [0.0] * len(documents))[: len(documents)]
    return [_to_float(s, 0.0) for s in scores]


def _score_by_api(query: str, documents: List[str]) -> List[float]:
    """
    在线 rerank API 打分（DashScope TextReRank，保留原实现）。
    :return: 与 documents 等长、下标对齐的 0~1 relevance_score
    """
    dashscope.api_key = reranker_config.text_rerank_api_key
    response = dashscope.TextReRank.call(
        model=reranker_config.text_rerank_model,
        query=query,
        documents=documents,
        top_n=len(documents),  # 全量返回：截断策略在 node_rerank 的断崖检测里做
        return_documents=False,
        instruct=reranker_config.text_rerank_instruct,
    )

    status_code = response.get("status_code") if hasattr(response, "get") else None
    if status_code != 200:
        message = response.get("message") if hasattr(response, "get") else response
        # 交给调用方（node_rerank）兜底成"不打分但可用的候选"，不在这里降级
        raise RuntimeError(f"DashScope rerank 调用失败(status={status_code}): {message}")

    output = response.output or {}
    results = output.get("results", []) if hasattr(output, "get") else []
    # 按 index 回填：响应里的 results 是打乱（按分数排序）的，
    # 必须用 index 还原成与入参 documents 同序，否则 zip 会张冠李戴
    scores = [0.0] * len(documents)
    for item in results or []:
        index = item.get("index")
        if index is None:
            continue
        index = int(index)
        if 0 <= index < len(documents):
            scores[index] = _to_float(item.get("relevance_score"), 0.0)
    return scores


def _to_float(value: Any, default: float = 0.0) -> float:
    """分数兜底转 float：库/接口可能返回 numpy 标量或 None，下游要做减法与阈值比较"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ==================== 对外统一入口 ====================

def rerank_documents(query: str, documents: Sequence[str]) -> List[float]:
    """
    统一重排序打分入口（保持 node_rerank 依赖的既有契约：返回与 documents 等长、
    下标对齐的分数列表）。按 reranker_config.rerank_mode 自动分流本地/在线。

    :param query: 用户问题（改写后的 rewritten_query）
    :param documents: 待打分的片段正文列表
    :return: list[float]，0~1 相关性分，顺序与 documents 一致
    """
    docs = _normalize_documents(documents)
    if not docs:
        # 空输入不能扔给模型（compute_score 内部 sentence_pairs[0] 会 IndexError），
        # 也不能让调用方拿到 None 去 zip
        return []
    if not any(d.strip() for d in docs):
        logger.warning("rerank 入参全部为空白文本，跳过模型调用直接返回 0 分")
        return [0.0] * len(docs)

    mode = _resolve_mode()
    if mode == MODE_API:
        return _score_by_api(query, docs)
    return _score_by_local(query, docs)


def rerank(query: str, documents: Sequence[str], top_n: int | None = None) -> List[Dict[str, Any]]:
    """
    统一重排序入口（"排序后的文档"形态，便于非 node_rerank 的调用方直接使用）。

    :param query: 用户问题
    :param documents: 候选文本列表
    :param top_n: 可选，只保留前 N 条（None 表示全部返回，截断由断崖检测负责）
    :return: [{"index": 原始下标, "document": 文本, "score": 0~1 相关性分}, ...]，按分数降序
    """
    docs = _normalize_documents(documents)
    scores = rerank_documents(query, docs)
    ranked = [{"index": i, "document": doc, "score": _to_float(score, 0.0)} for i, (doc, score) in
              enumerate(zip(docs, scores, strict=False))]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_n] if top_n else ranked


if __name__ == "__main__":
    # 手工自检入口：默认会触发本地模型加载（约 2.27GB，CPU 推理），主控串行验证时使用。
    # 想先验证配置与分支是否符合预期，可临时把 RERANK_MODE 设为 api 观察报错信息。
    _query = "HAK 180 烫金机如何调节转印温度？"
    _docs = [
        "转印温度的调节方式为：在操作面板上按 SET 键，默认建议设置在 110℃ 左右。",
        "今天中午去吃猪脚饭吧，这款机器的外观很漂亮。",
    ]
    logger.info(f"当前 rerank 模式：{_resolve_mode()}")
    for _doc, _score in zip(_docs, rerank_documents(_query, _docs), strict=False):
        logger.info(f"score={_score:.4f} doc={_doc[:30]}")
