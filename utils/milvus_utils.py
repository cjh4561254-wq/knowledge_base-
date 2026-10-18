# knowledge_base\utils\milvus_utils.py
"""Milvus 向量库封装层（集合建表 / 连接单例 / 稠密+稀疏混合检索）。

适配说明 —— 本机实际装的是 pymilvus **3.0.2**（服务端 Milvus 2.5.27），
课程文档与 7/29 参考版代码都是按 pymilvus 2.x 教程写的，这里有几处必须改，
否则要么 import/参数报错，要么静默行为不一致：

1. `MilvusClient.hybrid_search()` 3.x 的签名是
   `(collection_name, reqs, ranker, limit, output_fields, timeout, partition_names, **kwargs)`，
   **已经没有 `search_params` 形参**（2.x 教程里有）。继续传 `search_params=None`
   会掉进 `**kwargs` 并被原样转发到 `Prepare.search_requests_with_expr`，
   属于未定义行为（不同小版本可能直接 ParamError）。
   → 3.x 中每个向量字段的检索参数应写进各自 `AnnSearchRequest.param`
     （`{"metric_type": ..., "params": {"nprobe"/"ef": ...}}`），
     本层只在显式传入 search_params 时把它合并进 req.param，不再当顶层参数透传。
2. `create_schema()` / `prepare_index_params()` 在 3.x 是 **classmethod**，
   签名分别为 `(cls, **kwargs)`、`(cls, field_name="", **kwargs)`；
   `IndexParams.add_index(field_name, index_type="", index_name="", **kwargs)`
   仍可接 metric_type/params。所以建表写法本身兼容。
3. `WeightedRanker(*nums, norm_score=True)`：**3.x 默认值变成了 True**（2.x 是 False），
   所以 norm_score 必须由调用方显式传，不能依赖库默认，否则同一份代码在 2.x/3.x
   上排序结果不同。
4. `AnnSearchRequest` 的过滤参数 `expr` 仍保留（3.x 新增别名 `filter`，两者同时传会
   ParamError），故 `expr=expr` 写法兼容。
5. 返回结构：3.x `hybrid_search` 返回 `SearchResult → HybridHits → Hit(dict)`，
   每条命中是 `{主键名: pk, "distance": 融合分, "entity": {output_fields...}}`，
   **顶层 key 是主键字段名（kb_chunks 是 `chunk_id`，kb_item_names 是 `pk`），不是 `id`**。
   下游（node_rrf 取 `doc["entity"]["chunk_id"]`、node_item_name_confirm 取
   `hit["entity"]["item_name"]`）依赖固定结构，所以这里统一 normalize 成纯 dict，
   并同时给出 `id` 别名、保证 `entity` 内含主键，避免服务端不把主键放进 fields_data
   时下游拿到 None。

字段契约（与 node_import_milvus.py / node_item_name_recognition.py 写入端严格对齐）：
    kb_chunks      : chunk_id(pk) content title parent_title part file_title
                     item_name sparse_vector dense_vector
    kb_item_names  : pk           file_title item_name sparse_vector dense_vector
"""

import os
import threading
from collections.abc import Iterable, Sequence
from typing import Any

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker, WeightedRanker

from config.milvus_config import milvus_config
from tool.logger import logger

# ==================== 常量与配置 ====================

# 向量字段名：写入端（node_import_milvus / node_item_name_recognition）与检索端共用同一套名字
DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"
ITEM_NAME_FIELD = "item_name"
# 两个集合的主键字段名不同（kb_chunks 用 chunk_id，kb_item_names 用 pk）
CHUNKS_PK_FIELD = "chunk_id"
ITEM_NAME_PK_FIELD = "pk"

# 稠密向量度量：走配置（本机 MILVUS_METRIC_TYPE=COSINE），BGE-M3 的 dense 已 L2 归一化
_METRIC_TYPE = (os.getenv("MILVUS_METRIC_TYPE") or "COSINE").strip().upper()
# 稀疏向量度量：Milvus 对 SPARSE_FLOAT_VECTOR 只支持 IP（BGE-M3 词法权重按内积打分）
_SPARSE_METRIC_TYPE = "IP"
# 相似度阈值：供检索链路做低分过滤（本层只提供能力，默认不改 hybrid_search 行为，
# 因为混合检索的 distance 是加权融合分，和纯余弦分不同量纲，直接套用会误杀召回）
MIN_COSINE_SCORE = float(os.getenv("MILVUS_MIN_COSINE_SCORE") or "0.75")
# 向量维度：BGE-M3 固定 1024（EMBEDDING_DIM）
_VECTOR_DIM = int(os.getenv("EMBEDDING_DIM") or "1024")

# 客户端单例 + 锁：FastAPI/LangGraph 多线程并发检索时只建一条连接
_milvus_client: MilvusClient | None = None
_client_lock = threading.Lock()
# 已确认 load 成功的集合（进程内缓存，避免每次检索都发一次 RPC）
_loaded_collections: set[str] = set()
_loaded_lock = threading.Lock()


def get_milvus_client() -> MilvusClient:
    """
    获取 MilvusClient 单例（惰性创建）
    为什么用单例：MilvusClient 内部维护 gRPC 连接与元数据缓存，每个节点各建一次
    连接会在并发检索时打爆连接数；重复启动（幂等）也只是复用同一个对象。
    :return: 已连接的 MilvusClient 实例
    """
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client  # 已存在直接返回

    with _client_lock:
        # 双重检查：并发下只允许一个线程真正建连
        if _milvus_client is not None:
            return _milvus_client
        uri = milvus_config.milvus_url
        try:
            _milvus_client = MilvusClient(uri=uri)
        except Exception as e:  # 连不上必须原样上抛给调用方，这里只补日志定位（Milvus 地址/容器状态）
            logger.error(f"Milvus 连接失败 uri={uri}：{e!s}", exc_info=True)
            raise
        logger.info(f"Milvus 客户端已创建（单例）：{uri}")
        return _milvus_client


def escape_milvus_string(value: str) -> str:
    """
    Milvus数据库过滤表达式中字符串的安全转义函数（防止解析失败）
    作用：
        转义特殊字符（反斜杠、双引号），避免Milvus解析filter时报错
    参数：
        value: 需要转义的原始字符串
    返回：
        str: 转义后的安全字符串
    """
    if value is None:
        return ""
    # 转义反斜杠（\ → \\） 双引号（" → \"） 单引号（' → \'）
    value = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value


def build_in_filter(field_name: str, values: Iterable[Any] | None) -> str | None:
    """
    构造 `field in ["a","b"]` 形式的安全过滤表达式
    为什么需要：检索端目前用 f'item_name in {item_names}' 直接拼 Python list，
    生成的是单引号字面量，一旦商品名本身含引号（如 Brother's HAK180）表达式就会解析失败。
    :param field_name: 标量字段名（如 item_name）
    :param values: 取值集合
    :return: 过滤表达式；values 为空时返回 None（表示不加过滤条件）
    """
    items = [v for v in (values or []) if v is not None and str(v) != ""]
    if not items:
        return None
    quoted = ", ".join(f'"{_escape_for_double_quote(str(v))}"' for v in items)
    return f"{field_name} in [{quoted}]"


def _escape_for_double_quote(value: str) -> str:
    """
    双引号字面量专用转义：只转义 \\ 和 "，保留单引号原样。
    为什么不复用 escape_milvus_string：那个函数还会把 ' 转成 \\'，
    写入端用它拼 `file_title=='...'`（单引号字面量）是必须的；
    但在双引号字面量里 \\' 属于多余转义，个别 Milvus 版本会当非法转义序列解析失败。
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


# ==================== 建集合（幂等自动创建） ====================


def ensure_chunks_collection(
    client: MilvusClient | None = None,
    vector_dimension: int | None = None,
    collection_name: str | None = None,
) -> MilvusClient:
    """
    确保切片向量集合存在（不存在则建 Schema + 索引并 load），返回复用的客户端。
    幂等：重复启动时 has_collection 命中就直接返回，不会重复建表报错。
    字段 Schema 以本函数为唯一定义源，
    写入侧（node_import_milvus）与检索侧（node_search_embedding）都必须与此一致，
    否则插入会 DataNotMatch / 检索取不到字段。
    :param client: 可选，默认取单例
    :param vector_dimension: 可选，默认取 EMBEDDING_DIM（本机 1024）
    :param collection_name: 可选，默认 chunks_collection（kb_chunks）
    :return: MilvusClient 实例
    """
    client = client or get_milvus_client()
    name = collection_name or milvus_config.chunks_collection
    dim = int(vector_dimension or _VECTOR_DIM)

    if client.has_collection(name):
        _ensure_loaded(client, name)
        return client

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    # 主键：chunk_id 自增（node_import_milvus 插入后回填给下游做 RRF 去重键）
    schema.add_field(field_name=CHUNKS_PK_FIELD, datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片正文
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)  # 切片标题
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100, default_value="")
    schema.add_field(field_name="part", datatype=DataType.INT8)  # 分片编号
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)  # 源文件名（幂等删除依据）
    schema.add_field(field_name=ITEM_NAME_FIELD, datatype=DataType.VARCHAR, max_length=100)  # 商品名（过滤依据）
    schema.add_field(field_name=SPARSE_FIELD, datatype=DataType.SPARSE_FLOAT_VECTOR)  # BGE-M3 词法稀疏向量
    schema.add_field(field_name=DENSE_FIELD, datatype=DataType.FLOAT_VECTOR, dim=dim)  # BGE-M3 稠密向量

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=DENSE_FIELD,
        index_name="dense_vector_index",
        index_type="AUTOINDEX",
        metric_type=_METRIC_TYPE,
    )
    index_params.add_index(
        field_name=SPARSE_FIELD,
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type=_SPARSE_METRIC_TYPE,
        params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"},
    )

    # 3.x：带 index_params 的 create_collection 会自动 create_index + load_collection
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    _loaded_collections.add(name)
    logger.info(f"Milvus 集合已创建：{name}（dim={dim}, metric={_METRIC_TYPE}）")
    return client


def ensure_item_name_collection(
    client: MilvusClient | None = None,
    vector_dimension: int | None = None,
    collection_name: str | None = None,
) -> MilvusClient:
    """
    确保商品名向量集合存在（供 node_item_name_confirm 做主体名归一化检索）。
    字段与 node_item_name_recognition._create_item_name_collection 一致：
    主键是 pk（注意不是 chunk_id），所以本层 normalize 结果时不能硬编码主键名。
    :return: MilvusClient 实例
    """
    client = client or get_milvus_client()
    name = collection_name or milvus_config.item_name_collection
    dim = int(vector_dimension or _VECTOR_DIM)

    if client.has_collection(name):
        _ensure_loaded(client, name)
        return client

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name=ITEM_NAME_PK_FIELD, datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name=ITEM_NAME_FIELD, datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name=DENSE_FIELD, datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name=SPARSE_FIELD, datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=DENSE_FIELD,
        index_name="dense_vector_index",
        index_type="IVF_FLAT",
        metric_type=_METRIC_TYPE,
        params={"nlist": 128},
    )
    index_params.add_index(
        field_name=SPARSE_FIELD,
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type=_SPARSE_METRIC_TYPE,
        params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"},
    )

    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    _loaded_collections.add(name)
    logger.info(f"Milvus 集合已创建：{name}（商品名集合, dim={dim}）")
    return client


def _ensure_loaded(client: MilvusClient, collection_name: str) -> None:
    """
    确保集合已加载到内存（Milvus 未 load 的集合检索会直接报 collection not loaded）。
    进程内缓存一次即可：load_collection 是幂等 RPC，但没必要每次检索都发。
    """
    with _loaded_lock:
        if collection_name in _loaded_collections:
            return
        try:
            client.load_collection(collection_name)
            _loaded_collections.add(collection_name)
        except Exception as e:  # noqa: BLE001  # 已加载/权限类失败不影响检索，降级为 warning
            logger.warning(f"Milvus 集合[{collection_name}] load 失败（可能已加载）：{e!s}")


# ==================== 混合检索 ====================


def create_hybrid_search_requests(
    dense_vector,
    sparse_vector,
    dense_params=None,
    sparse_params=None,
    expr=None,
    limit=5,
    dense_field: str = DENSE_FIELD,
    sparse_field: str = SPARSE_FIELD,
):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量（list[float]，长度须等于建表 dim）
    :param sparse_vector: 文本生成的稀疏向量（dict[int, float]，3.x AnnSearchRequest 原生支持）
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据；None/空串表示不过滤
    :param limit: 单向量搜索返回结果数量，默认5
    :param dense_field: 稠密向量字段名（两个集合同名，保留参数便于复用）
    :param sparse_field: 稀疏向量字段名
    :return: 搜索请求列表，包含[dense_req, sparse_req]
    """
    # 稠密向量默认搜索参数：COSINE（度量类型可配，必须与建表索引一致，否则服务端报错）
    if dense_params is None:
        dense_params = {"metric_type": _METRIC_TYPE}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": _SPARSE_METRIC_TYPE}

    # expr 为空串/None 时统一成 None：3.x 里 expr 与 filter 同时给会 ParamError，
    # 且非字符串的 expr 会抛 DataTypeNotMatchException
    filter_expr = expr if (expr is not None and str(expr).strip()) else None

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[_as_dense_list(dense_vector)],  # 数据（稠密向量），必须是 [[float...]]
        anns_field=dense_field,  # 要查的字段 名称
        param=dense_params,  # 相似度规则（3.x：nprobe/ef 等写在 param 的 "params" 子键）
        expr=filter_expr,
        limit=limit,  # 召回条数
    )

    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[_as_sparse_dict(sparse_vector)],
        anns_field=sparse_field,
        param=sparse_params,
        expr=filter_expr,
        limit=limit,
    )
    # 配置参数
    return [dense_req, sparse_req]


def hybrid_search(
    client,
    collection_name,
    reqs,
    ranker_weights=(0.5, 0.5),
    norm_score=False,
    limit=5,
    output_fields=None,
    search_params=None,
    ranker: str | Any | None = None,
    min_score: float | None = None,
):
    """
    执行Milvus稠密+稀疏向量混合搜索
    基于WeightedRanker实现双向量搜索结果加权融合，提升检索准确性
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
    :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
                       （注意：3.x 的 WeightedRanker 默认值已变成 True，这里必须显式传）
    :param limit: 混合搜索最终返回结果数量，默认5
    :param output_fields: 需要返回的字段列表，默认返回item_name
    :param search_params: 额外检索参数（如 {"nprobe": 16} / {"ef": 64}）。
                          3.x 的 hybrid_search 没有这个形参，这里合并进每个
                          AnnSearchRequest.param["params"]，不再当顶层参数透传。
    :param ranker: 融合策略："weighted"(默认，按 ranker_weights)/"rrf"(RRFRanker)，
                   也可直接传入构造好的 Ranker 对象
    :param min_score: 可选，按 distance 做低分过滤；None 表示不过滤（保持参考版行为）。
                      需要 MILVUS_MIN_COSINE_SCORE 语义过滤时由调用方显式传入。
    :return: 归一化后的结果 list[list[dict]]（外层对应每个查询向量），搜索失败返回None
    """
    try:
        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = [ITEM_NAME_FIELD]

        ranker_obj = _build_ranker(ranker, ranker_weights, norm_score)
        _merge_search_params(reqs, search_params)

        _ensure_loaded(client, collection_name)

        res = _call_hybrid_search(client, collection_name, reqs, ranker_obj, limit, output_fields)

        # 转成项目内部统一结构（纯 dict + id 别名 + entity 内含主键）
        normalized = _normalize_hits(res)
        if min_score is not None:
            normalized = [[hit for hit in one_query if (hit.get("distance") or 0.0) >= min_score]
                          for one_query in normalized]

        total = sum(len(one_query) for one_query in normalized)
        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{total}条结果")
        return normalized
    except Exception as e:  # noqa: BLE001  # 检索层对上游的约定是「失败返回 None」，不能把异常抛穿 LangGraph
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{e!s}", exc_info=True)
        return None


def _call_hybrid_search(client, collection_name, reqs, ranker_obj, limit, output_fields):
    """
    真正调用 3.x 的 client.hybrid_search
    只传 3.x 签名里存在的参数（collection_name/reqs/ranker/limit/output_fields）。
    额外兜底：若进程内缓存认为集合已加载、但实际被其它进程 release 掉，
    清缓存重新 load 后重试一次（Milvus「集合未加载」是最常见的偶发检索错误）。
    """
    try:
        return client.hybrid_search(
            collection_name=collection_name,  # 集合名称 表
            reqs=reqs,  # 请求
            ranker=ranker_obj,  # 使用排序器
            limit=limit,  # 返回条数限制
            output_fields=output_fields,  # 输出的内容
        )
    except Exception as e:  # 只拦「集合未加载」，其余异常交给外层统一记录并返回 None
        if "not loaded" not in str(e).lower():
            raise
        logger.warning(f"集合[{collection_name}]未加载，重新加载后重试混合检索")
        with _loaded_lock:
            _loaded_collections.discard(collection_name)
        _ensure_loaded(client, collection_name)
        return client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=ranker_obj,
            limit=limit,
            output_fields=output_fields,
        )


def _build_ranker(ranker, ranker_weights: Sequence[float], norm_score: bool):
    """
    构造融合排序器
    为什么单独拆出来：2.x/3.x 的 Ranker 都是普通对象，但 WeightedRanker 在 3.x
    改成了 `(*nums, norm_score=True)`，权重必须是位置参数且个数与 reqs 一致。
    """
    if ranker is not None and not isinstance(ranker, str):
        return ranker
    mode = (ranker or "weighted").strip().lower()
    if mode in ("rrf", "rrfranker"):
        # RRF 只看名次不看分数量纲，天然规避稠密(COSINE 0~1)与稀疏(IP 0~几十)的量级差
        return RRFRanker(k=60)
    if mode not in ("weighted", "weightedranker"):
        raise ValueError(f"不支持的 ranker 类型：{ranker}")
    weights = list(ranker_weights or (0.5, 0.5))
    return WeightedRanker(*weights, norm_score=bool(norm_score))


def _merge_search_params(reqs, search_params) -> None:
    """
    把 2.x 风格的 search_params 合并到各 AnnSearchRequest.param["params"]
    （3.x 没有顶层 search_params 形参，per-request 参数才有效）。
    兼容两种写法：{"nprobe": 16} 与 {"params": {"nprobe": 16}, "metric_type": "COSINE"}。
    """
    if not search_params:
        return
    # param 是只读 property，但返回的是内部字典本身 → 原地 update 即可生效
    inner_extra = {
        k: v for k, v in search_params.items() if k not in ("metric_type", "anns_field", "params", "limit")
    }
    for req in reqs or []:
        param = req.param
        if not isinstance(param, dict):
            continue
        merged = dict(param.get("params") or {})
        merged.update(search_params.get("params") or {})
        merged.update(inner_extra)
        param["params"] = merged
        if "metric_type" in search_params:
            param["metric_type"] = search_params["metric_type"]


def _normalize_hits(res) -> list[list[dict]]:
    """
    把 pymilvus 3.x 的 SearchResult/HybridHits/Hit 归一成 list[list[dict]]：
        [{"id": pk, "<主键字段名>": pk, "distance": float, "entity": {含主键在内的输出字段}}]
    为什么必须做：
      1) 下游（node_rrf / node_item_name_confirm）用 doc["entity"][...] 取值，
         而服务端不保证把主键放进 entity —— 缺失时补进去，否则 RRF 用 chunk_id 做键会全部命中 None；
      2) Hit 是 dict 子类且带懒加载向量的私有实现，写进 Mongo / json.dumps 前转纯 dict 更稳；
      3) 顶层统一加 "id" 别名，兼容课程里 hit["id"] 的老写法。
    """
    if not res:
        return []
    normalized: list[list[dict]] = []
    for one_query in res:
        hits: list[dict] = []
        for hit in one_query or []:
            raw = dict(hit)
            entity = dict(raw.get("entity") or {})
            pk_name, pk_value = _pick_pk(raw, entity)
            if pk_name and entity.get(pk_name) is None and pk_value is not None:
                entity[pk_name] = pk_value
            one_hit: dict[str, Any] = {
                "id": pk_value,
                "distance": _as_float(raw.get("distance")),
                "entity": entity,
            }
            if pk_name:
                one_hit[pk_name] = pk_value
            # 保留服务端附加信息（highlight/offset），有就带上，没有不影响下游
            for extra_key in ("highlight", "offset"):
                if extra_key in raw:
                    one_hit[extra_key] = raw[extra_key]
            hits.append(one_hit)
        normalized.append(hits)
    return normalized


def _as_float(value) -> float | None:
    """distance 兜底转 float：服务端偶发返回 numpy 标量/None，下游要做阈值比较必须是原生 float"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_pk(raw: dict, entity: dict):
    """
    推断主键：3.x 的 Hit 用真实主键字段名做顶层 key（chunk_id / pk），
    因此取第一个非元数据 key；退化情况用 "id"，再从 entity 兜底。
    """
    meta_keys = {"id", "distance", "entity", "offset", "highlight"}
    for key, value in raw.items():
        if key not in meta_keys:
            return key, value
    if "id" in raw:
        return "id", raw["id"]
    for candidate in (CHUNKS_PK_FIELD, ITEM_NAME_PK_FIELD):
        if candidate in entity:
            return candidate, entity[candidate]
    return None, None


def _as_dense_list(vector) -> list:
    """稠密向量兜底转换：numpy/元组 → list[float]（Milvus 要求与建表 dim 等长的浮点列表）"""
    if vector is None:
        raise ValueError("dense_vector 不能为空")
    if hasattr(vector, "tolist"):
        return vector.tolist()
    if isinstance(vector, (list, tuple)) and len(vector) == 1 and isinstance(vector[0], (list, tuple)):
        # 容错：调用方误传 [[...]]（已经是 batch 形式）
        return list(vector[0])
    return [float(x) for x in vector]


def _as_sparse_dict(vector) -> dict:
    """
    稀疏向量兜底转换：Milvus 接受 {int: float}，也接受 [(idx, weight), ...]。
    flagembedding 返回的 key 可能是 numpy.int64，统一转成原生 int，避免 gRPC 序列化报错。
    """
    if vector is None:
        raise ValueError("sparse_vector 不能为空")
    if isinstance(vector, dict):
        return {int(k): float(v) for k, v in vector.items()}
    if hasattr(vector, "tocoo"):  # scipy 稀疏矩阵：(row, col, data) 三元组
        coo = vector.tocoo()
        return {int(idx): float(w) for idx, w in zip(coo.col, coo.data, strict=False)}
    return {int(idx): float(weight) for idx, weight in vector}


if __name__ == "__main__":
    # 只读连通性自检：不建集合、不写数据（写入验证由主控串行执行）
    _client = get_milvus_client()
    logger.info(f"Milvus 连接成功，当前集合：{_client.list_collections()}")
