# processor/query_processor/state.py

from typing import TypedDict


class QueryGraphState(TypedDict):
    """
    查询流程图状态
    包含整个查询流程中传递的所有数据。

    公共契约：检索侧 6 个节点 + web/api/query_service.py 都在读写这些 key，
    禁止改名/删除；新增字段必须先在此声明（未声明的 key 会被 LangGraph 丢弃）。
    并行三节点各自只返回自己字段的增量 dict（如 {"embedding_chunks": [...]}），
    因此不需要 reducer；串行节点返回全量 state，每步只有一个写入方，同样安全。
    """

    session_id: str  # 会话ID
    message_id: str  # 消息ID

    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # 已向量化的假设性问题切片
    hyde_doc: str  # LLM 生成的假设性答案原文（node_search_embedding_hyde 写入，
                   # 供排查召回质量；未声明的 key 会被 LangGraph 静默丢弃，必须在此登记）
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: list[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题   理解用户的意图
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出