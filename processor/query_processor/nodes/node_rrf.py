# processor/query_processor/nodes/node_rrf.py
"""RRF（Reciprocal Rank Fusion，倒数排名融合）粗排节点。

职责：把「向量检索」和「HyDE 检索」两路召回结果按排名融合成一份统一候选，
交给下游 node_rerank 做精排。网络搜索（web_search_docs）没有 chunk_id、
量纲也不同，按课程实现是由 node_rerank 直接合并参与精排的，不在这里融合。

为什么只看排名不看原始分数：向量检索是 COSINE 分（0~1）、HyDE 走的是加权融合分、
网搜只有名次，三者不可直接相加；RRF 的 1/(k+rank) 把它们统一到同一量纲。
"""

import hashlib
from typing import Any, Dict, List, Tuple

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.task_utils import add_done_task

# -----------------------------
# RRF 融合全局常量
# -----------------------------
# 平滑常数 k：课程文档、RRF 原论文（Manning 2008）与 Milvus RRFRanker 的默认值都是 60。
# k 越大越"公平"（头部名次差被压平），k=0 会退化成赢家通吃，故必须用 60。
RRF_K: int = 60
# 融合后送给精排的候选条数（粗排 TopN）。
# 为什么取 5 而不是 3：下游断崖检测的下界 RERANK_MIN_TOPK=3、上界 10，
# 粗排只给 3 条时候选数 ≤ 下界，断崖检测永远没有可截断的区间，等价于失效；
# 多给几条也正是"两阶段检索"里召回池该比最终 TopK 大的意义。
RRF_MAX_RESULTS: int = 5
# 各路权重（都取 1.0 即经典 RRF；后续要主体名加权只改这里）
RRF_WEIGHT_VECTOR: float = 1.0
RRF_WEIGHT_HYDE: float = 1.0


def _first_nonempty(*values: Any) -> str:
    """
    取第一个非空文本（None / 空白串都算「没取到」），统一返回 str。
    用于 title → file_title → parent_title 的来源兜底链，避免下游拿到 None 再 str(None) 打成 "None"。
    """
    for value in values:
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        if text.strip():
            return text
    return ""


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象，读取 embedding_chunks / hyde_embedding_chunks
        :return: 更新后的状态对象，写入 rrf_chunks（永远是 list，下游可直接遍历）
        """

        try:
            # 1. 获取各搜索路的搜索结果（上游节点失败时字段可能是 None，这里统一兜底）
            embedding_search_list = self._step_1_flatten(state.get("embedding_chunks"), "向量检索")
            hyde_embedding_search_list = self._step_1_flatten(state.get("hyde_embedding_chunks"), "HyDE检索")

            # 2. 构建融合前的数据结构：(该路文档列表, 该路权重)
            rrf_inputs = [
                (embedding_search_list, RRF_WEIGHT_VECTOR),
                (hyde_embedding_search_list, RRF_WEIGHT_HYDE),
            ]

            # 3. 根据RRF公式进行数据的融合
            rrf_merge_results = self._rrf_merge(rrf_inputs, k=RRF_K, max_results=RRF_MAX_RESULTS)

            # 4. 将分数清洗出去（下一个步骤重新排序，不参考这个步骤的分数）
            state["rrf_chunks"] = [doc for doc, _ in rrf_merge_results]

            logger.info(
                f"RRF 融合完成：向量路 {len(embedding_search_list)} 条 + HyDE 路 {len(hyde_embedding_search_list)} 条 "
                f"→ 去重后取 Top{RRF_MAX_RESULTS}，实际 {len(state['rrf_chunks'])} 条"
            )
        except Exception as e:  # noqa: BLE001  # 融合层对下游的约定是「给出一份可用候选」，任何异常都降级为空列表而非抛穿 LangGraph
            logger.exception(f"RRF 融合失败，降级为空候选: {e!s}")
            state["rrf_chunks"] = []

        # 正常与异常路径都要打点（否则前端进度条会一直停在"倒排融合 运行中"）
        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        return state

    def _step_1_flatten(self, raw_hits: Any, route_name: str) -> List[Dict[str, Any]]:
        """
        把一路召回结果压平成统一的文档 dict 列表（去重键、正文字段都在这一步定型）。

        上游（node_search_embedding / node_search_embedding_hyde）的字段结构由
        utils/milvus_utils._normalize_hits() 决定：
            [{"id": pk, "chunk_id": pk, "distance": float, "entity": {chunk_id, content, item_name, ...}}, ...]
        注意 pymilvus 3.x 的命中顶层 key 是**主键字段名 chunk_id**，不是 2.x 的 "id"，
        所以这里 chunk_id 的取值顺序是 entity → 顶层 chunk_id → 顶层 id，任一能拿到即可。
        """
        docs: List[Dict[str, Any]] = []
        for hit in raw_hits or []:
            doc = self._normalize_hit(hit)
            if doc is not None:
                docs.append(doc)

        if not docs:
            logger.warning(f"RRF 输入：{route_name} 路无有效结果")
        return docs

    @staticmethod
    def _normalize_hit(hit: Any) -> Dict[str, Any] | None:
        """
        单条命中 → 扁平文档 dict。
        返回 None 表示这条不是可融合的结构（非 dict / entity 不是 dict）。
        """
        if not isinstance(hit, dict):
            return None

        entity = hit.get("entity")
        if isinstance(entity, dict):
            doc: Dict[str, Any] = dict(entity)
        elif entity is None:
            # 兼容直接就是扁平文档的调用方（例如本文件的单测 mock、其它检索源）
            doc = dict(hit)
        else:
            return None

        # 向量字段对精排/答案组装毫无用处，还会让 state 体积暴涨、Mongo 序列化失败
        for vector_key in ("dense_vector", "sparse_vector", "emb", "embedding"):
            doc.pop(vector_key, None)

        # chunk_id：entity 里最准；服务端没把主键放进 entity 时（pymilvus 3.x 的常见情况）
        # 退回命中顶层的 chunk_id / id
        if doc.get("chunk_id") is None:
            doc["chunk_id"] = hit.get("chunk_id", hit.get("id"))

        # 正文统一成 str：Web 源用 snippet、部分源用 text，缺失时留空串由去重键兜底
        content = doc.get("content")
        if content is None:
            content = doc.get("text") or doc.get("snippet")
        doc["content"] = content if isinstance(content, str) else ("" if content is None else str(content))

        # 来源标题兜底链：切片标题 → 源文件名 → 父标题 → 商品主体名
        # （kb_chunks 的来源/身份字段，Schema 见 utils/milvus_utils.ensure_chunks_collection；
        #  全项目不存在 doc_id，别按 doc_id 取值）。
        # 为什么在这里兜而不是留空串往下传：title 是 node_rerank 组装 reranked_docs、
        # node_answer_output 组 prompt 元数据头与引用来源 sources 的展示字段，
        # 空标题 = 前端答案里的引用没有名字。判「非空」而不只是判 None，
        # 是因为导入侧对缺元数据的切片会写空串（node_import_milvus.prepare_insert_rows）。
        doc["title"] = _first_nonempty(
            doc.get("title"), doc.get("file_title"), doc.get("parent_title"), doc.get("item_name")
        )

        return doc

    def _rrf_merge(self, rrf_inputs, k: int = RRF_K, max_results: int | None = None) -> List[Tuple[Dict, float]]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法
        :param rrf_inputs: 输入数据列表，每个元素是一个元组，包含一个列表和一个权重
        :param k: RRF中的平滑参数
        :param max_results: 最大结果数量限制，如果是None，则获取所有结果
        :return: 融合后的结果列表 [({},得分),({},得分),...]
        """

        # 1. 遍历需要融合的数据，并组装数据
        chunk_data = {}  # 最后数据
        chunk_scores = {}  # 综合得分
        missing_key_count = 0  # 拿不到主键的条数（只告警一次，避免刷屏）
        for route_index, (rrf_input, weight) in enumerate(rrf_inputs):
            # 对当前这一路计算
            for rank, doc in enumerate(rrf_input or [], start=1):
                if not isinstance(doc, dict):
                    logger.warning(f"RRF 跳过非字典元素（路 {route_index}, 名次 {rank}）: {type(doc)}")
                    continue

                chunk_id = self._doc_key(doc)
                if chunk_id is None:
                    missing_key_count += 1
                    continue

                # 计算每个文档的得分，并将得分组织在list中
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank)
                # 将所有文档组织在list中
                chunk_data.setdefault(
                    chunk_id,
                    doc
                )  # 只保留第一次遇到的那个文档实体，不会覆盖。“如果 chunk_id 这个键还不存在，就把它存进字典，值设为 doc；如果已经存在了，什么也不做。”

        if missing_key_count:
            logger.warning(f"RRF 有 {missing_key_count} 条结果既无 chunk_id 也无正文，已跳过融合")

        # 2. 将得分和文档组装在一起
        unsorted_results = [(chunk_data[chunk_id], score) for chunk_id, score in chunk_scores.items()]

        # 3. 按照得分降序排列
        sorted_results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)
        # topN  取前几条
        return sorted_results[:max_results] if max_results else sorted_results

    @staticmethod
    def _doc_key(doc: Dict[str, Any]):
        """
        取去重键：主键 chunk_id（含 2.x 的 id 别名）优先；实在拿不到就用正文指纹。

        为什么不能直接用 doc.get("chunk_id") 当键：None 也是合法的字典键，
        一旦这批结果全没有 chunk_id（例如 output_fields 漏配、换了 pymilvus 版本），
        所有文档会被并进同一个 None 键 → RRF 只剩 1 条结果，而且是静默错误。
        用正文 hash 兜底能把「同内容」继续去重、把「不同内容」保持区分，同时显式告警。
        """
        chunk_id = doc.get("chunk_id")
        if chunk_id is None:
            chunk_id = doc.get("id")
        if chunk_id is not None:
            return chunk_id

        content = (doc.get("content") or "").strip()
        if not content:
            return None
        digest = hashlib.md5(content[:500].encode("utf-8", errors="ignore")).hexdigest()[:16]
        logger.warning(f"RRF 结果缺少 chunk_id，改用正文指纹去重（title={doc.get('title')}）: {content[:30]}...")
        return f"anon:{digest}"


if __name__ == '__main__':

    # 模拟两路检索结果
    mock_state = {
        # 直接硬搜得到相似度高的片段
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        # 通过假设性文档搜到的相似度高的片段   比如 chunk_1 两路排名里面都高
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#3"}},
        ]
    }

    node_rrf = NodeRrf()  # __call__  process
    result = node_rrf(mock_state)
    logger.info(serialize_json(result, indent=4))  # 打印日志
