# processor/query_processor/prompt/item_name_confirm.py
# 商品名确认节点的提示词。
#
# 契约说明（改这里必须同步 nodes/node_item_name_confirm.py::_step_4_extract_info）：
# - 模板使用「命名占位符」{history_text} 与 {query}，节点侧用
#   ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=..., query=...) 关键字参数填充，
#   两者必须一一对应（历史项目出现过 {} 位置占位与 format 参数错位数不匹配的 bug）。
# - JSON 示例中的花括号必须写成 {{ }}（format 转义后输出为普通 { }），
#   否则 .format() 会把单花括号当占位符解析直接抛 KeyError/IndexError。

# 系统提示词（人设）：明确"只输出 JSON"，从源头减少围栏/解释性文字等脏输出
ITEM_NAME_EXTRACT_SYSTEM_PROMPT = (
    "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"
    "你必须严格按照用户要求的JSON格式输出："
    "只输出一个合法的JSON对象，不输出任何解释性文字，不使用Markdown代码块包裹。"
)

# 用户提示词
ITEM_NAME_EXTRACT_TEMPLATE = """
历史会话：
{history_text}
当前用户问题：
{query}

请根据历史会话和当前问题，提取用户正在询问的商品名称（item_names）。
1. 如果用户明确提到了商品名称，请提取出来。可能有一个或多个，但不能重复。
2. 如果用户使用了代词（如"这个"、"它"），请结合历史会话指代消解，确定商品名称；
   如果历史会话中正在讨论某个商品，当前追问也应提取该商品名称。
3. 如果无法确定商品名称，item_names 返回空列表 []，不要臆造商品名。
4. 请重新改写用户的问题（rewritten_query），使其成为包含商品名称的独立完整问题；
   rewritten_query 必须是非空字符串，无法改写时原样返回当前用户问题。

输出要求（必须遵守）：
- 只输出一个JSON对象，不要输出JSON以外的任何内容。
- 不要使用 ```json 之类的代码块包裹。
- JSON必须合法可解析：双引号、无尾逗号。

输出schema（字段名和类型固定）：
{{
    "item_names": ["商品A", "商品B"],
    "rewritten_query": "关于商品A和商品B，..."
}}
"""
