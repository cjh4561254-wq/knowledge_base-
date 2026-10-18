# processor/query_processor/prompt/search_embedding_hyde.py
"""HyDE 假设性文档生成提示词。

约定：模板有且仅有 1 个占位符 `{rewritten_query}`，与调用方
`HYDE_PROMPT.format(rewritten_query=...)`（node_search_embedding_hyde._step_1_create_hyde_doc）
严格一一对应 —— 正文中禁止再出现裸的花括号，否则 str.format 会把它们当占位符解析并抛 KeyError。
"""

HYDE_PROMPT = """
请基于以下用户查询生成一个简洁的回答范文。
用户查询: {rewritten_query}
要求：
1. 回答要简洁明了，包含核心信息即可
2. 假设你是该领域的专家，提供专业的解释
3. 不要使用"假设"、"可能"等不确定的词汇
4. 保持回答与查询主题高度相关
5. 使用中文回答且不超过300字
6. 直接给出结论与操作步骤说明（含设备名、参数名、数值单位等专业关键词），不要复述问题、不要写前言结语
"""
