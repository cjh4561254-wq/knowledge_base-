# processor/query_processor/nodes/node_search_embedding_hyde.py
"""HyDE（Hypothetical Document Embeddings）检索节点（三路召回的第 2 路）。

思路：短问题语义稀疏，直接拿去向量检索召回差；先让 LLM「脑补」一段假设性答案，
把「问题 + 假设文档」一起向量化再去 Milvus 召回，用更饱满的语义信号桥接 Query/Document 的鸿沟。

降级要求（重要）：假设文档为空、LLM 调用异常、`.format()` 出错时，**必须退回用原始问题检索**，
而不是让节点崩 —— 三路召回是 langgraph 并行分支，这一路崩会整图失败。
检索本体复用 node_search_embedding.hybrid_recall()，两条链路的 output_fields/量纲/阈值保持一致。
"""

from processor.query_processor.base import NodeBase
from processor.query_processor.nodes.node_search_embedding import (
    RETURN_LIMIT,
    hybrid_recall,
)
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.llm_utils import get_llm_client
from utils.task_utils import add_done_task


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        HyDE (Hypothetical Document Embedding) 检索节点
        核心思想：通过LLM生成假设性答案（HyDE文档），将其向量化后用于检索，以解决短查询语义稀疏问题。

        执行步骤：
        1. 参数提取：从会话状态中获取改写后的查询（rewritten_query）和已确认的商品名（item_names）。
        2. 生成假设文档 (Step 1)：调用LLM，基于用户问题生成一段假设性的理想回答（即HyDE文档）。
        3. 混合检索 (Step 2)：
           - 将“用户问题 + 假设文档”合并，生成BGE-M3稠密+稀疏向量。
           - 在Milvus中执行混合检索（带商品名过滤），召回最相似的知识切片。
        4. 结果封装：返回检索到的切片列表和生成的假设文档，更新会话状态。

        :param state: 会话状态字典，包含 rewritten_query, item_names 等
        :return: 包含 hyde_embedding_chunks (检索结果) 和 hyde_doc (假设文档) 的字典
        """
        # 1、用户问题和已确认商品名
        rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
        item_names = state.get("item_names")

        try:
            # 2、生成假设性文档（失败/为空时内部已降级为空串，不抛异常）
            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)
            logger.info(f"hyde_doc假设性回答: {hyde_doc}")

            # 3、用“重写问题 + 假设文档”检索切片
            res = self._step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names,
            )
            # 4、结果封装（空结果也返回空列表，结构与另一路完全一致）
            return {
                "hyde_embedding_chunks": res,
                "hyde_doc": hyde_doc,
            }

        except Exception as e:
            logger.exception(f"假设性文档向量搜索失败: {e}")
            return {"hyde_embedding_chunks": [], "hyde_doc": ""}
        finally:
            # 打点：done 放 finally，保证异常/降级路径也会把这一路从 running 里摘掉
            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))

    def _step_2_search_embedding_hyde(
            self,
            rewritten_query: str,
            hyde_doc: str,
            item_names=None
    ):
        """
        阶段2：利用“重写问题 + 假设性文档”生成 embedding，并到向量库检索切片。

        :param rewritten_query: 改写后的查询
        :param hyde_doc: Step 1 生成的假设性文档（为空时自动退化为「只用原始问题」检索）
        :param item_names: 商品名称列表，用于元数据过滤 (item_name in [...])
        :return: 检索结果列表
        """
        # 1、拼接查询与假设文档，形成更丰富的语义上下文
        # 这里把用户问题 + 假设答案拼在一起生成向量，相当于：
        # 既保留了用户的原始意图（rewritten_query）
        # 又增强了语义丰富度（hyde_doc）
        combined_text = f"{rewritten_query or ''} {hyde_doc or ''}".strip()

        # 2、降级路径：假设文档缺失/生成失败 → 不抛异常，直接用原始问题做这一路检索，
        #    这样这一路仍然有投票权，RRF 不会因为 HyDE 失败而少一路
        if not hyde_doc or not str(hyde_doc).strip():
            logger.warning("步骤2: 假设文档为空或生成失败，降级为原始问题向量检索")
            combined_text = (rewritten_query or "").strip()

        if not combined_text:
            logger.warning("步骤2: 问题与假设文档均为空，本路返回空结果")
            return []

        # 3、向量化 + 混合检索 + 余弦阈值过滤（与 node_search_embedding 走同一份实现，
        #    output_fields 同样取 node_search_embedding.RECALL_OUTPUT_FIELDS =
        #    ["chunk_id","content","title","parent_title","part","file_title","item_name"]
        #    （+ 临时 dense_vector，取回后即剔除），
        #    与 WeightedRanker(0.8,0.2, norm_score=True)，保证三路结果结构可被 node_rrf 直接合并）
        #    注意：这里绝不另写一份 output_fields —— 历史上 HyDE 路与向量路字段不一致，
        #    导致 HyDE 召回的切片在 node_rerank/node_answer_output 里恒无标题（引用来源空白）。
        #    共用 hybrid_recall 即从根上保证两路都带 title/parent_title/file_title。
        return hybrid_recall(
            query_text=combined_text,
            item_names=item_names,
            top_k=RETURN_LIMIT,
            log_prefix="步骤2: ",
        )

    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        """
        阶段1：利用大模型根据用户查询生成假设性文档（Hypothetical Document）。
        HyDE的核心在于：利用LLM生成一个“虚构但相关”的文档，用该文档的向量去检索真实的文档，
        从而缓解短查询（Query）与长文档（Document）在语义空间不匹配的问题。

        :param rewritten_query: 用户改写后的查询语句
        :return: LLM生成的假设性文档内容；生成失败或内容为空时返回空串（由调用方降级）
        """
        logger.info("步骤1: 开始生成假设性文档")

        if not rewritten_query or not str(rewritten_query).strip():
            logger.warning("步骤1: 问题为空，不生成假设文档")
            return ""

        try:
            # LLM 走 config.lm_config（本机 DeepSeek：deepseek-flash / deepseek-v4-pro）
            llm = get_llm_client()
            # HYDE_PROMPT 只含 1 个占位符 {rewritten_query}，与此处唯一实参严格对应
            hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
            hyde_doc = llm.invoke(hyde_prompt).content
            if not hyde_doc or not str(hyde_doc).strip():
                logger.warning("步骤1: LLM 返回的假设文档为空，降级为原始问题检索")
                return ""
            return str(hyde_doc).strip()

        except Exception as e:
            # 不 raise：HyDE 只是加分项，失败必须降级而不是拖垮整条检索链
            logger.exception(f"步骤1: 生成假设文档失败，降级为原始问题检索: {e}")
            return ""


if __name__ == "__main__":
    init_state = {
        "session_id": "test_search_hyde_001",
        "is_stream": False,
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机", "BrotherHAK180烫金扫描仪"],
    }
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    result = node_search_embedding_hyde(init_state)
    logger.info(serialize_json(result, indent=4))
