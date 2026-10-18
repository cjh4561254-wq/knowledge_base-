# processor/query_processor/nodes/node_search_embedding.py
"""向量库混合检索节点（三路召回的第 1 路）。

本文件同时提供两路向量检索（本节点 + HyDE 节点 node_search_embedding_hyde）共用的
混合召回实现 `hybrid_recall()`，避免两条链路在「过滤表达式构造 / output_fields /
ranker 量纲 / 相似度阈值」上各写一份而慢慢跑偏。

关键契约（改动前先看 utils/milvus_utils.py 的模块注释）：
1. 本机 pymilvus 3.0.2：`client.hybrid_search()` 已无 `search_params` 形参，
   `WeightedRanker` 的 `norm_score` 默认值从 2.x 的 False 变成 True
   → 本层必须显式传，否则同一份代码在 2.x/3.x 上排序结果不同。
2. `hybrid_search()` 的返回值已被 `_normalize_hits()` 规整为
   `[{"id": pk, "<主键字段名=chunk_id>": pk, "distance": float, "entity": {...}}]`，
   **顶层主键 key 是 `chunk_id`（不是 `id`）**，且 `entity` 内一定含主键；
   下游 node_rrf 就是按 `doc["entity"]["chunk_id"]` 消费，故本节点原样透传 hit 列表。
3. 商品名过滤一律走 `build_in_filter()`：课程里 `f'item_name in {item_names}'` 直接拼
   Python list，字面量是单引号，商品名一旦含单引号（如 Brother's HAK180）Milvus 解析表达式失败。
4. `RECALL_OUTPUT_FIELDS` 必须覆盖下游全部所需字段（课程参考版只取
   `["chunk_id","content","item_name"]`，导致 rerank 与引用来源拿不到标题）：
   `kb_chunks` 的真实身份/来源字段是
   `chunk_id(主键) + content + title + parent_title + part + file_title + item_name`
   （见 `utils/milvus_utils.ensure_chunks_collection`，全项目不存在 `doc_id`），
   其中 title 取不到时由 node_rrf / node_rerank 按 `title → file_title → parent_title` 兜底。
"""

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import serialize_json
from utils.milvus_utils import (
    DENSE_FIELD,
    MIN_COSINE_SCORE,
    build_in_filter,
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from utils.task_utils import add_done_task

# 稠密/稀疏融合权重：语义（dense）为主、词法（sparse）为辅（课程规格 0.8/0.2）
RANKER_WEIGHTS = (0.8, 0.2)
# 单路向量检索的候选池大小：RRF 融合与后续 rerank 都靠足够的候选才有效，故多召回一些
CANDIDATE_LIMIT = 10
# 本路对外返回的 Top-K（课程规格：底层召回 10 条，本路只交给下游 5 条，最终条数由 rerank 断崖截断）
RETURN_LIMIT = 5

# 检索需要取回的业务字段：必须与 utils/milvus_utils.ensure_chunks_collection 建表字段一致
# - chunk_id：node_rrf 的融合/去重主键（缺它整路会被算成同一条）
# - content/title：node_rerank 送 reranker 打分、node_answer_output 组 prompt
# - parent_title/file_title/part：答案引用溯源（定位到源文件与章节）
# - item_name：主体名，用于校验过滤是否生效
# - DENSE_FIELD：只为在本地算「真实余弦相似度」做阈值过滤，取回后立刻从 entity 里剔除，
#   绝不让 1024 维向量进入 state / Mongo（否则每条结果多几 KB 且 json 序列化会炸）
RECALL_OUTPUT_FIELDS = [
    "chunk_id",
    "content",
    "title",
    "parent_title",
    "part",
    "file_title",
    "item_name",
    DENSE_FIELD,
]

# entity 里必须齐备的「文本型」来源字段。
# 为什么要再补一层键：Milvus 只对「集合里真实存在且有值」的字段回键，
# 一旦某批数据是旧 Schema / 字段整键缺失，下游 node_rrf 展平后拿不到 title，
# 一路传到 node_answer_output 的引用来源 sources[N]["title"] 就是空（前端引用无标题）。
# 检索侧统一把缺失键补成空串，让「缺字段」与「字段为空」在下游是同一种情况，
# 由 node_rrf / node_rerank 的 title → file_title → parent_title → item_name 兜底链取值。
RECALL_TEXT_FIELDS = ("title", "parent_title", "file_title", "item_name")


def ensure_recall_fields(entity: dict) -> dict:
    """
    补齐单条命中 entity 的来源字段（原地修改并返回，不覆盖已有值）。

    :param entity: hybrid_search 规整结果里的 hit["entity"]
    :return: 同一个 dict，缺失的文本字段被补成空串
    """
    for field in RECALL_TEXT_FIELDS:
        if entity.get(field) is None:
            entity[field] = ""
    return entity


def _cosine(dense_a: list, dense_b: list) -> float | None:
    """
    计算两个向量的余弦相似度。
    BGE-M3 的 dense 向量在模型内部已做 L2 归一化（normalize_embeddings=True），
    因此点积即余弦，这里仍除一次模长以防上游改了归一化配置。

    :return: 余弦值；向量缺失/长度不一致等无法判断的情况返回 None（调用方选择「不误杀」）
    """
    try:
        if not dense_a or not dense_b or len(dense_a) != len(dense_b):
            return None
        dot = sum(x * y for x, y in zip(dense_a, dense_b, strict=False))
        norm_a = sum(x * x for x in dense_a) ** 0.5
        norm_b = sum(y * y for y in dense_b) ** 0.5
        if not norm_a or not norm_b:
            return 0.0  # 零向量就是「毫无相似可言」，给 0 分而不是 None
        return dot / (norm_a * norm_b)
    except (TypeError, ValueError) as e:
        logger.warning(f"计算余弦相似度失败，跳过阈值过滤: {e}")
        return None


def filter_by_cosine(hits: list, query_dense: list, threshold: float) -> list:
    """
    按「查询向量 vs 切片向量」的真实余弦相似度做低分过滤，并把取回的 dense_vector 剔掉。

    为什么不用 hybrid_search(min_score=...)：
      混合检索顶层 `distance` 是 WeightedRanker 的加权融合分
      （稠密 COSINE 0~1 与稀疏 IP 几十量纲混在一起），拿它和 MILVUS_MIN_COSINE_SCORE=0.75
      直接比是量纲错误，会大面积误杀召回（utils/milvus_utils.py 顶部注释同观点）。
      这里改为现场用双向量点积算真余弦，语义与建表 metric_type=COSINE 完全一致。

    兜底策略：如果全部候选都低于阈值，仍然保留分最高的一条并告警 ——
    三路召回是并行分支，某路整路为空等于这一路对 RRF 没有投票，
    阈值应当「降权」而不是「灭路」。

    :param hits: hybrid_search 返回的单路命中列表（每项含 entity）
    :param query_dense: 查询文本的稠密向量
    :param threshold: 最小余弦相似度（来自 MILVUS_MIN_COSINE_SCORE）
    :return: 过滤后的命中列表（entity 内已不含向量字段，并写入 entity["score"]）
    """
    scored: list = []
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        entity = hit.get("entity")
        if not isinstance(entity, dict):
            entity = {}
            hit["entity"] = entity
        chunk_dense = entity.pop(DENSE_FIELD, None)  # 弹出即从结果里移除，不进 state
        # 来源字段补齐（title/parent_title/file_title/item_name 缺键 → 空串），
        # 保证下游 node_rrf 展平后一定能拿到「可展示的来源」而不是 KeyError/None
        ensure_recall_fields(entity)
        cos_score = _cosine(query_dense, chunk_dense) if chunk_dense else None
        if cos_score is None:
            # 取不到向量（老数据/字段缺失）时不做判断，保留该条：宁可多召回不误杀
            entity.setdefault("score", None)
            scored.append((None, hit))
        else:
            entity["score"] = round(float(cos_score), 4)
            scored.append((float(cos_score), hit))

    kept = [hit for score, hit in scored if score is None or score >= threshold]

    # 全部被阈值拦下：保留最接近的一条，避免这一路彻底为空
    if hits and not kept:
        best = max(scored, key=lambda x: (x[0] is not None, x[0] or -1.0), default=None)
        if best is not None:
            logger.warning(
                f"本路 {len(hits)} 条召回的余弦相似度均低于阈值 {threshold}，"
                f"保留最高分 {best[0]} 的一条以避免整路为空（下游 rerank 会再精排）"
            )
            best[1]["entity"]["score"] = best[0] if best[0] is not None else None
            kept = [best[1]]
    return kept


def hybrid_recall(
    query_text: str,
    item_names: list | None,
    top_k: int = RETURN_LIMIT,
    log_prefix: str = "",
) -> list:
    """
    稠密 + 稀疏混合检索的公共实现（本节点与 HyDE 节点共用）。

    流程：文本向量化 → 安全过滤表达式 → create_hybrid_search_requests
          → hybrid_search（WeightedRanker 显式 norm_score）→ 余弦阈值过滤 → Top-K

    :param query_text: 参与向量化的文本（本节点=改写后问题；HyDE=问题+假设文档）
    :param item_names: 已确认商品名列表，构造 item_name in [...] 过滤；为空则全库检索
    :param top_k: 本路对外返回的条数
    :param log_prefix: 日志前缀，区分两条链路
    :return: 命中列表（每项 {"id","chunk_id","distance","entity":{...}}），异常/无结果返回空列表
    """
    try:
        if not query_text or not str(query_text).strip():
            logger.warning(f"{log_prefix}检索文本为空，跳过向量检索")
            return []

        # 1、生成向量 (Dense + Sparse)：generate_embeddings 契约 -> {"dense": [[1024]...], "sparse": [{int:float}...]}
        embeddings = generate_embeddings([query_text])
        dense_vec = embeddings["dense"][0]
        sparse_vec = embeddings["sparse"][0]

        # 2、商品名过滤：build_in_filter 生成双引号字面量并转义，商品名含引号也不会让表达式解析失败
        expr = build_in_filter("item_name", item_names)
        if expr:
            logger.info(f"{log_prefix}过滤条件: {expr}")
        else:
            logger.info(f"{log_prefix}未指定商品名过滤，将全库检索")

        # 3、构造 Milvus 混合搜索请求（dense=COSINE / sparse=IP 的度量由 utils 层按建表配置给出）
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            expr=expr,
            limit=CANDIDATE_LIMIT,  # 单向量召回条数：多留候选给 RRF/rerank 挑选
        )

        # 4、执行混合检索
        logger.info(f"{log_prefix}开始执行 Milvus 混合检索，集合: {milvus_config.chunks_collection}")
        res = hybrid_search(
            client=get_milvus_client(),
            collection_name=milvus_config.chunks_collection,
            reqs=reqs,
            ranker_weights=RANKER_WEIGHTS,
            norm_score=True,  # 3.x 默认 True、2.x 默认 False：必须显式锁定量纲（归一化后加权）
            limit=CANDIDATE_LIMIT,  # 融合后先取回候选池，再按余弦阈值筛
            output_fields=RECALL_OUTPUT_FIELDS,
            min_score=None,  # 融合分量纲不等于余弦，阈值过滤交给 filter_by_cosine
        )

        # 5、结果规整：utils 层返回 list[list[dict]]（外层对应每个查询向量），本层只查一条文本 → 取 res[0]
        hits = res[0] if res else []

        # 6、真实余弦阈值过滤（MILVUS_MIN_COSINE_SCORE）
        filtered = filter_by_cosine(hits, dense_vec, MIN_COSINE_SCORE)

        logger.info(f"{log_prefix}向量检索完成：候选 {len(hits)} 条，阈值后 {len(filtered)} 条，返回 {min(len(filtered), top_k)} 条")
        return filtered[:top_k]

    except Exception as e:
        # 这一路失败只返回空列表：三路召回并行，抛穿会让整张图（含 RRF）一起崩
        logger.exception(f"{log_prefix}向量检索失败: {e}")
        return []


class NodeSearchEmbedding(NodeBase):
    """
   节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
   """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
        流程：用户问题向量化 → 构造带商品名过滤的混合搜索请求 → 执行稠密+稀疏混合检索 → 返回检索结果
        :param state: Dict - 会话状态字典，包含上游传递的核心信息，关键字段：
                      {
                          "rewritten_query": str,   # step4改写后的完整用户问题（含商品名）
                          "item_names": list[str],  # step7已确认的标准化商品名列表
                      }

        :return: Dict - 检索结果字典，仅包含embedding_chunks字段，供下游节点使用：
                 {
                     "embedding_chunks": List[Dict]  # Milvus检索结果列表，无结果则为空列表
                                                     # 每个元素为一条匹配的向量数据，含业务字段
                 }
        """
        # 1、用户问题和已确认商品名
        query = state.get("rewritten_query") or state.get("original_query") or ""
        item_names = state.get("item_names")

        try:
            # 2、混合检索（公共实现：向量化 → 过滤 → 召回 → 余弦阈值）
            chunks = hybrid_recall(
                query_text=query,
                item_names=item_names,
                top_k=RETURN_LIMIT,
                log_prefix="[向量检索] ",
            )
            # 3、构造并返回结果：无结果也要返回空列表，保证 node_rrf 拿到一致结构
            return {"embedding_chunks": chunks}
        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {"embedding_chunks": []}
        finally:
            # 打点：无论成功/失败/空结果都要 done，否则并行时这一路永远挂在 running，前端进度条不动
            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))


if __name__ == "__main__":
    init_state = {
        "session_id": "test_search_embedding_001",
        "is_stream": False,
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金", "BrotherHAK-180烫金机", "BrotherHAK180烫金扫描仪"],
    }
    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)
    logger.info(serialize_json(result, indent=4))  # 打印
