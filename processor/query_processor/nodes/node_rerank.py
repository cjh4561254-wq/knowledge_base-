# processor/query_processor/nodes/node_rerank.py
"""Rerank 精排节点：Cross-Encoder 打分 + 断崖检测动态 TopK 截断。

流程：合并多源候选（RRF 粗排结果 + 网络搜索）→ 统一入口 rerank_documents()
（按配置走本地 bge-reranker-v2-m3 或在线 API）→ 按分数降序 → 断崖检测截断。

输出 state["reranked_docs"]（字段名不可改，下游 node_answer_output 依赖），每条：
    {"content": str, "title": str, "chunk_id": int|str|None, "url": str|None,
     "source": "local"|"web", "score": float|None}
title 恒为 str（不会再出现恒空串）：本地路按 title → file_title → parent_title 兜底，
网络路按 网页标题 → entity 来源名 → url 兜底；三条都取不到才是空串。
"""

import re
from typing import Any, Dict, List

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.reranker_http_utils import rerank_documents
from utils.task_utils import add_done_task

# -----------------------------
# Rerank / TopK 全局常量
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 3  # 总数最少条数

# 断崖阈值（绝对，判断高分文档）
# 注意量纲：分数必须是 0~1（在线 API 的 relevance_score / 本地 normalize=True 的 sigmoid），
# 若换回未归一化的 logits，这个绝对阈值会恒定成立、导致只保留 1 条。
RERANK_GAP_ABS: float = 0.5
# 断崖阈值（相对，判断低分文档）
RERANK_GAP_RATIO: float = 0.25

# 来源标题兜底链的候选键（本地切片在 kb_chunks 里的来源/身份字段，
# Schema 见 utils/milvus_utils.ensure_chunks_collection：title/parent_title/file_title/item_name，
# 全项目不存在 doc_id）。
# 顺序说明：切片标题最具体 → 源文件名（同一篇文档的引用通常同名）
# → 父标题 → 商品主体名（最后兜底，至少引用能显示"是哪台设备"，好过空白）。
TITLE_SOURCE_KEYS: tuple = ("title", "file_title", "parent_title", "item_name")
# Markdown 标题前缀（node_document_split 的 title 直接取 MD 标题行，会带 "## "）
_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")


def _strip_heading_marks(text: str) -> str:
    """剥掉 Markdown 标题的 # 前缀：引用来源/ prompt 元数据头给人看，不该带 "## "。"""
    return _MD_HEADING_RE.sub("", text).strip()


def _resolve_title(doc: Dict[str, Any]) -> str:
    """
    从一条候选文档里取「可展示的来源标题」。

    为什么要兜底链：node_answer_output 的引用来源 sources[N]["title"] 与 prompt 元数据头的
    [title=...] 都取这里的 title。历史上检索侧 output_fields 没带 title（课程参考版只取
    chunk_id/content/item_name），title 就恒为空串，前端答案里的引用标题全空。
    取值顺序：扁平键 → entity 子字典（网络搜索路的来源信息放在 entity 里），
    每个字典内再按 TITLE_SOURCE_KEYS 取第一个非空值。
    取不到时返回空串（绝不返回 None，避免下游拼出 "None"）。

    :param doc: rrf_chunks 的扁平文档 或 web_search_docs 的原始文档
    :return: str 标题（可能为空串）
    """
    candidates = [doc]
    entity = doc.get("entity")
    if isinstance(entity, dict):
        candidates.append(entity)

    for source in candidates:
        for key in TITLE_SOURCE_KEYS:
            text = _as_text(source.get(key)).strip()
            if text:
                return _strip_heading_marks(text)

    # 网络结果的兜底：Tavily 偶发无 title 时用 url 充当来源名，至少引用可溯源
    url_text = _as_text(doc.get("url")).strip()
    return url_text


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        执行重排序
        流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
        :param state: 需包含 rrf_chunks、web_search_docs、rewritten_query
        :return: 更新后的 state，包含 reranked_docs
        """

        # 1. 合并多源文档（这一步不依赖模型，但异常也要兜住，否则下面连兜底数据都没有）
        try:
            merged_multi_docs: List[Dict[str, Any]] = self._step_1_merge_multi_source_docs(state)
        except Exception as e:  # noqa: BLE001  # 合并是纯内存操作，失败说明上游 state 结构变了：记录并退化成空候选
            logger.exception(f"Rerank 合并多源文档失败: {e!s}")
            merged_multi_docs = []

        try:
            # 2. Rerank 精排(精排打分)
            reranked_docs: List[Dict[str, Any]] = self._step_2_rerank_merged_docs(state, merged_multi_docs)

            # 3. 动态 Top_K 截取(断崖检测)
            cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)

            # 4. 更新state
            state['reranked_docs'] = cutoff_docs
        except Exception as e:  # noqa: BLE001  # 精排/截断失败必须降级而不是抛穿：下游 answer 节点要保证拿到 list 上下文
            # 退回「未打分但已合并好的候选」，按 RRF 名次取最小条数，
            # 让 answer 节点仍有上下文（分数为 None 时 prompt 里不会打 score 标签）
            logger.exception(f"Rerank 节点异常，降级为未打分的 RRF 候选: {e!s}")
            state['reranked_docs'] = [
                {**doc, "score": None} for doc in merged_multi_docs[: RERANK_MIN_TOPK]
            ]

        # 正常与异常路径都要打点（否则前端进度条永远卡在"重排序 运行中"）
        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        # 5. 返回state
        return state

    def _step_3_cliff_cutoff(self, ranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断。"""
        # 防御性 编程
        if not ranked_docs:
            return []

        # 上边界和下边界
        # 上边界
        upper_bound = min(RERANK_MAX_TOPK, len(ranked_docs))

        # 下边界
        lower_bound = min(RERANK_MIN_TOPK, upper_bound)

        # 最后的结果要 3~4条  都可以

        # 分数全部缺失（精排失败走了降级路径）时无法做断崖判断：
        # 此时宁少勿滥，只按 RRF 名次保留最小条数，避免把一堆不确定相关的文档塞给 LLM
        if not any(_as_float(doc.get("score")) is not None for doc in ranked_docs):
            logger.warning(f"Rerank 分数缺失，跳过断崖检测，按 RRF 名次保留前 {lower_bound} 条")
            return ranked_docs[:lower_bound]

        # 默认值：取满硬上限（最多10条）
        # 初始值 cutoff_pos=4
        cutoff_pos = upper_bound

        # 遍历范围：从min_topk-1到max_topk-2（索引从0开始），检测相邻两个文档的分数差
        # 例：min_topk=3，max_topk=10 → 遍历i=2,3,4,5,6,7,8（对应第3~9条文档，检测与下一条的差距）
        for idx in range(lower_bound - 1, upper_bound - 1):
            current_score = _as_float(ranked_docs[idx].get("score"))  # 取出当前分数
            next_score = _as_float(ranked_docs[idx + 1].get("score"))  # 取出下一个分数

            # 防御性编程
            if current_score is None or next_score is None:
                continue

            # 计算相邻文档的分数绝对差距（因已降序，gap≥0）
            abs_gap = current_score - next_score
            # 计算相对差距：绝对差距 / 当前文档分数（+1e-6避免除数为0/极小值，防止程序报错）
            # 1e-6 是 Python 中科学计数法的写法，等价于 0.000001（10 的负 6 次方，也就是百万分之一）。
            rel_gap = abs_gap / (abs(current_score) + 1e-6)
            # 拿两个绝对值之间 比值
            # 触发断崖截断条件：绝对差距≥绝对阈值 OR 相对差距≥相对阈值
            # 满足任一条件，说明下一条文档相关性骤降，截断在当前位置
            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO:
                # 最终取前i+1条（索引转实际数量，如i=2 → 取前3条）
                cutoff_pos = idx + 1
                logger.debug(f"断崖检测: 位置 {idx + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}")
                break

        logger.info(f"断崖检测截断：候选 {len(ranked_docs)} 条 → 保留 {cutoff_pos} 条")
        return ranked_docs[:cutoff_pos]

    def _step_2_rerank_merged_docs(self, state: QueryGraphState, merged_multi_docs: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """使用 Reranker 模型对文档进行精排"""

        # 候选为空（RRF 没结果且网搜也没结果）：直接返回空列表，
        # 不能把空 query/空文档喂给模型（本地分支 compute_score 对空 list 会 IndexError）
        if not merged_multi_docs:
            logger.warning("Rerank 候选文档为空，跳过精排")
            return []

        try:
            user_query = state.get('rewritten_query') or state.get('original_query') or ""
            if not str(user_query).strip():
                logger.warning("Rerank 查询为空，跳过精排（按 RRF 名次保留）")
                return [{**doc, "score": None} for doc in merged_multi_docs]

            # 获取文档列表的conten字段组成列表
            contents = [doc.get("content") for doc in merged_multi_docs]
            # 调用Rerank模型：交叉编码器（精排阶段）
            # Query 和 Document 联合编码，精度更高
            # 调重排序模型 给他 问题  +  所有的 片段  给我  分数
            rerank_scores = rerank_documents(user_query, contents)
            # rerank_scores 问题和所有片段的 综合得分  精排

            # 分数与文档必须严格等长，否则 zip 会静默丢文档（少一条就少一条上下文）
            rerank_scores = _align_scores(rerank_scores, len(merged_multi_docs))

            scored_docs = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]
            # 等同如下写法
            # scored_docs = []
            # for doc, score in zip(merged_multi_docs, rerank_scores):
            #     scored_docs.append({
            #         "content": doc.get("content"),
            #         "title": doc.get("title"),
            #         "chunk_id": doc.get("chunk_id"),
            #         "url": doc.get("url"),
            #         "source": doc.get("source"),
            #         "score": float(score),
            #     })

            sorted_score_docs = sorted(
                scored_docs,
                # 兜底 -inf：万一某条分数缺失，排序时沉到末尾而不是抛 TypeError
                key=lambda x: x["score"] if x["score"] is not None else float("-inf"),
                reverse=True
            )

            return sorted_score_docs

        except Exception as e:  # noqa: BLE001  # 模型/接口任何失败都降级为「未打分候选」，由断崖检测按 RRF 名次兜底截断
            logger.exception(f"Rerank 重排序失败: {e!s}")
            # 原实现在这里是 `[{**merged_multi_docs, "score": None}]`：
            # 对 list 做 ** 解包会直接 TypeError，等于异常里再抛一次异常、把整个图打断。
            # 正确降级是「逐条保留文档本体 + score=None」，交给断崖检测按 RRF 名次截断。
            return [{**doc, "score": None} for doc in merged_multi_docs]

    def _step_1_merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 结果和网络搜索结果为统一格式"""

        final_docs = []

        # 1. 获取本地 RRF 的文档  rrf_chunks向量融合的结果
        # 上游节点失败时字段可能是 None，直接 for ... in None 会 TypeError
        for rrf_doc in _as_list(state.get('rrf_chunks')):
            if not isinstance(rrf_doc, dict):
                logger.warning(f"跳过非法 RRF 文档（非 dict）: {type(rrf_doc)}")
                continue
            content = _as_text(rrf_doc.get('content'))
            if not content.strip():
                # 正文为空/纯空白绝不能进候选：下游 answer 节点拼 prompt 是 "标签\n" + content，
                # content=None 会 TypeError，空白串则是在浪费 LLM 上下文
                logger.warning(f"跳过无正文的本地切片: chunk_id={rrf_doc.get('chunk_id')}")
                continue
            format_rrf_doc = {
                "content": content,
                # title 走兜底链（title → file_title → parent_title），本地切片标题缺失时
                # 仍能给出源文件名，下游引用来源不会是空串
                "title": _resolve_title(rrf_doc),
                "chunk_id": rrf_doc.get('chunk_id', rrf_doc.get('id')),
                "url": None,
                "source": "local"  # 本地
            }
            final_docs.append(format_rrf_doc)

        # 2. 获取 web 远程的文档
        for web_doc in _as_list(state.get('web_search_docs')):
            if not isinstance(web_doc, dict):
                logger.warning(f"跳过非法网络搜索结果（非 dict）: {type(web_doc)}")
                continue
            content = _as_text(web_doc.get('snippet')).strip() or _as_text(web_doc.get('content')).strip()
            if not content:
                logger.warning(f"跳过无正文的网络搜索结果: url={web_doc.get('url')}")
                continue
            format_web_doc = {
                "content": content,
                # 网络路的 title 是网页标题（Tavily/MCP 返回），与本地路的章节标题同为「来源名称」，
                # 语义一致、字段同名即可（前端靠 source=local/web 区分二者，不会互相覆盖）；
                # 网页无标题时兜底链会退到 entity.file_title(=标题或URL) → url
                "title": _resolve_title(web_doc),
                "chunk_id": None,
                "url": web_doc.get('url'),
                "source": "web"
            }
            final_docs.append(format_web_doc)

        if not final_docs:
            logger.warning("Rerank 合并后无任何候选文档（RRF 与网搜都为空）")
        return final_docs


def _as_list(value: Any) -> List[Any]:
    """把 state 字段兜底成可遍历的 list（None / 单条 dict 都能处理）"""
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def _as_text(value: Any) -> str:
    """文本字段统一成 str：None → 空串，非字符串 → str()，避免下游字符串拼接报错"""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _as_float(value: Any) -> float | None:
    """分数兜底：只有能转成 float 的才参与断崖判断，否则返回 None（该位跳过比较）"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _align_scores(scores: Any, doc_count: int) -> List[float | None]:
    """
    保证「分数列表」与「文档列表」严格等长且下标对齐。
    为什么：打分实现（本地模型/在线接口）返回条数若与入参不一致，
    zip 会静默截断，候选文档会在没人报警的情况下消失。
    """
    if not isinstance(scores, list):
        logger.warning(f"rerank 返回类型异常({type(scores)})，全部按无分数处理")
        return [None] * doc_count
    if len(scores) != doc_count:
        logger.warning(f"rerank 返回 {len(scores)} 个分数，候选有 {doc_count} 条，按下标对齐补齐")
        scores = list(scores) + [None] * max(0, doc_count - len(scores))
        scores = scores[:doc_count]
    return [_as_float(s) for s in scores]


if __name__ == "__main__":
    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "title": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "title": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)
    logger.info(serialize_json(result, indent=4))
