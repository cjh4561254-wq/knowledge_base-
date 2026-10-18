# utils/json_format_utils.py

"""
JSON 格式化工具模块

提供统一的 JSON 序列化和格式化功能，确保项目中 JSON 输出的一致性
"""


from typing import Any, Dict, Optional, Tuple
from bson import ObjectId
import ast
import json
import re
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID

class CustomJSONEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，支持 MongoDB ObjectId 等特殊类型
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, cls=CustomJSONEncoder)

class MongoEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，支持MongoDB ObjectId等特殊类型的序列化

    支持的类型：
    - ObjectId: 转换为字符串
    - datetime/date: 转换为ISO格式字符串
    - Decimal: 转换为浮点数
    - UUID: 转换为字符串
    """

    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


def serialize_json(data, ensure_ascii=False, indent=None, **kwargs):
    """
    将数据序列化为JSON字符串，自动处理MongoDB ObjectId等特殊类型

    参数：
        data: 要序列化的数据（字典、列表等）
        ensure_ascii: 是否确保ASCII编码（默认False，支持中文）
        indent: 缩进空格数（默认None，紧凑格式；设为4可美化输出）
        **kwargs: 其他传递给json.dumps的参数

    返回：
        JSON格式的字符串

    示例：
        # 基础用法
        json_str = serialize_json({"_id": ObjectId("..."), "name": "测试"})

        # 美化输出
        json_str = serialize_json(data, indent=4)

        # 在日志中使用
        logger.info(serialize_json(result, indent=2))
    """
    return json.dumps(data, cls=MongoEncoder, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def to_json_file(data, filepath, ensure_ascii=False, indent=4, **kwargs):
    """
    将数据序列化为JSON文件，自动处理MongoDB ObjectId等特殊类型

    参数：
        data: 要序列化的数据
        filepath: 文件保存路径
        ensure_ascii: 是否确保ASCII编码（默认False，支持中文）
        indent: 缩进空格数（默认4，美化输出）
        **kwargs: 其他传递给json.dump的参数

    示例：
        to_json_file(result, "output.json")
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=MongoEncoder, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


# ======================================================================
# 脏 JSON 修复层：把大模型返回的"不完美" JSON 修成可解析的 Python 结构
# ======================================================================
# 大模型输出 JSON 的常见脏形态（课程 day12/day20 的节点里只处理了第一种）：
#   1. ```json ... ``` 代码块包裹，甚至前后还带解释性文字
#   2. 尾逗号：{"a": [1, 2,], }
#   3. 单引号风格（Python 字面量）：{'item_names': ['HAK180']}
#   4. max_tokens 截断：字符串没闭合、括号栈没弹完
#   5. true/false/null 与 True/False/None 混用
# 修复策略：从"最像原文"到"最激进补救"逐级尝试，任何一级解析成功立即返回，
# 全部失败则返回调用方给的 default——保证返回值类型始终符合调用方契约，
# 调用方永远不会拿到 None 之后还要判类型（None 本身也是一种合法的 default 选择）。

# 代码围栏：```json / ```JSON / ```，允许围栏前后有空白或紧跟内容
_FENCE_RE = re.compile(r"```\s*(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_LONE_FENCE_RE = re.compile(r"```\s*(?:json|JSON)?\s*", re.DOTALL)
# 尾随逗号：, 后面紧跟 } 或 ]
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _try_json(text: str) -> Tuple[bool, Any]:
    """尝试标准 JSON 解析，返回 (是否成功, 结果)。"""
    try:
        return True, json.loads(text)
    except (ValueError, TypeError):
        return False, None


def _try_literal_eval(text: str) -> Tuple[bool, Any]:
    """
    尝试 Python 字面量解析（处理单引号字典/列表、True/False/None）。
    ast.literal_eval 只接受字面量语法，不会执行任意代码，安全。
    注意：它不接受 JSON 的 true/false/null，所以仅作为单引号形态的兜底。
    """
    try:
        obj = ast.literal_eval(text)
        # 只接受容器类型，避免把 "123" 这类碎片解析成裸数字误导调用方
        if isinstance(obj, (dict, list)):
            return True, obj
        return False, None
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return False, None


def _strip_fences(text: str) -> str:
    """去掉 ```json 代码块包裹；优先取成对围栏内的内容，只有半个围栏时裸删标记。"""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return _LONE_FENCE_RE.sub("", text).replace("```", "").strip()


def _json_span(text: str) -> str:
    """
    截取第一个 '{' 或 '[' 开始的片段（丢弃 LLM 爱在 JSON 前输出的"好的，结果是："等前置文本）。
    找不到起始括号则原样返回，交给后续步骤报错走 default 分支。
    """
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text
    return text[min(starts):]


def _close_truncated(fragment: str) -> str:
    """
    给被截断的 JSON 片段补全闭合：
    1. 逐字符扫描（跳过字符串内部），维护 { [ 的期望闭合栈；
    2. 结尾停在字符串内部 → 去掉悬挂的反斜杠后补引号；
    3. 补引号后若残留 ' , ' 或 ' : ' 尾巴（如 {"a": 截断）先删掉；
    4. 按栈逆序补齐 ] }。
    只做结构补全、不猜内容；补完仍解析失败会继续尝试"退到上一个逗号截断点"的激进方案。
    """
    stack = []          # 期望的闭合符栈
    in_str = False
    quote_ch = ""
    esc = False
    for ch in fragment:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
                continue
            elif ch == quote_ch:
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str = True
                quote_ch = ch
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]":
                # 只有类型匹配才弹栈，避免脏文本把栈弹穿
                if stack and stack[-1] == ch:
                    stack.pop()

    fixed = fragment
    if esc:
        # 结尾停在转义符上（...jpg\），补引号前必须删掉这个孤立反斜杠
        fixed = fixed[:-1]
    if in_str:
        fixed += '"' if quote_ch == '"' else "'"
    # 补齐闭合前，去掉会让结构非法的 dangling 逗号/冒号
    fixed = fixed.rstrip()
    while fixed and fixed[-1] in ",:":
        fixed = fixed[:-1]
    return fixed + "".join(reversed(stack))


def _repair_candidates(text: str):
    """
    按"侵入性从低到高"生成候选修复串，逐个尝试解析。
    层级设计成短路式：越靠前越接近原文，保证可解析的正常输出绝不会被改坏。
    """
    span = _json_span(_strip_fences(text))
    no_comma = _TRAILING_COMMA_RE.sub(r"\1", span)
    # 截断补救时额外尝试"退回到最后一个逗号"再闭合，处理 {"a":1, "b": [1,2,"xx 这种尾巴残缺形态
    trunc_variants = []
    closed = _close_truncated(no_comma)
    trunc_variants.append(closed)
    cut = no_comma.rfind(",")
    if cut > 0:
        trunc_variants.append(_close_truncated(no_comma[:cut]))
    cutbrace = max(no_comma.rfind("}"), no_comma.rfind("]"))
    if cutbrace > 0:
        trunc_variants.append(_close_truncated(no_comma[:cutbrace + 1]))

    for candidate in (span, no_comma, *trunc_variants):
        candidate = candidate.strip()
        if not candidate:
            continue
        # 每一级都先试严格 JSON，再试 Python 字面量（单引号/True/None 形态）
        yield ("json", candidate)
        yield ("literal", candidate)


def repair_json(raw: Any, default: Any = None, expect: Optional[type] = None) -> Any:
    """
    把大模型返回的脏 JSON 文本修复并解析为 Python 结构。

    参数：
        raw:     LLM 原始输出（str；已是 dict/list 时原样返回，None/坏类型返回 default）
        default: 全部修复手段失败后的兜底值。调用方应传入与自身契约一致的类型，
                 例如 node_item_name_confirm 场景传 {"item_names": [], "rewritten_query": q}
        expect:  期望的容器类型（dict 或 list）；解析成功但类型不符时同样落 default，
                 保证调用方永远拿到期望类型，不会因 LLM 返回标量/字符串而抛 AttributeError

    返回：
        解析成功的 dict/list，或 default。
    """
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw if (expect is None or isinstance(raw, expect)) else default
    if not isinstance(raw, str) or not raw.strip():
        return default

    for kind, candidate in _repair_candidates(raw):
        ok, obj = _try_json(candidate)
        if not ok and kind != "json":
            ok, obj = _try_literal_eval(candidate)
        if ok and isinstance(obj, (dict, list)):
            if expect is None or isinstance(obj, expect):
                return obj
            return default
    return default


def repair_item_names_result(raw: Any, original_query: str = "") -> Dict[str, Any]:
    """
    面向 node_item_name_confirm._step_4_extract_info 返回契约的专用修复入口。

    契约（见该节点源码 docstring）：返回
        {"item_names": List[str], "rewritten_query": str}
    这里把「解析失败时的兜底返回值」直接做成该契约的合法值，节点拿到后
    无需再做任何 None 判断即可安全访问两个字段。

    参数：
        raw:            LLM 返回的原始文本（可能带 ```json 包裹/尾逗号/单引号/截断）
        original_query: 用户原始问题，改写缺失时用它回填
    返回：
        含 item_names（字符串列表，已去空白）与 rewritten_query（str）的 dict
    """
    result = repair_json(raw, default={}, expect=dict)
    if not isinstance(result, dict):
        result = {}

    item_names = result.get("item_names")
    if not isinstance(item_names, list):
        # LLM 偶尔把单个商品名写成字符串，兼容收编成列表
        item_names = [item_names] if isinstance(item_names, str) and item_names.strip() else []
    result["item_names"] = [str(name).strip() for name in item_names if str(name).strip()]

    rewritten = result.get("rewritten_query")
    result["rewritten_query"] = rewritten if isinstance(rewritten, str) and rewritten.strip() else original_query
    return result
