# processor/import_processor/nodes/node_item_name_recognition.py
"""主体（商品名）识别节点。

职责：从「文档切片」中识别出这份文档所属的标准商品名，回填到 state 与每个切片，
并把 商品名 + 稠密/稀疏向量 写入 Milvus 的 kb_item_names 集合，供检索链路的
`node_item_name_confirm` 做「用户口中的名字 → 库内标准名字」语义对齐。

本机适配说明（为什么和课程/参考版代码不一样）：
1. 模型档位：主体识别是高频、低难度的抽取任务，走便宜档 `lm_config.item_model`
   （本机 = deepseek-flash），不再用 `llm_model`（deepseek-v4-pro）。
   该账号只有 deepseek-flash / deepseek-v4-pro 两个模型，写死 deepseek-chat 等
   老名字会直接 404。
2. 取值容错：deepseek-flash 可能把正文放进 reasoning 相关字段，
   统一走 `_extract_llm_text()`（content → 内容块 → reasoning_content → thinking），
   并且「全空」时抛 LLMError，绝不把空字符串当成识别结果写进库/state。
3. 脏输出容错：prompt 要求返回 JSON，但小模型经常返回 ```json 包裹/尾逗号/
   单引号/被 max_tokens 截断/干脆一句大白话。这里复用
   `utils.json_format_utils.repair_json`（只读，不改它），并在它也无法还原成
   容器时退回「按纯文本取第一行有效商品名」的兜底路径。
4. 建表：不再在本文件重复写一份 Schema（原实现把 dim 硬编码成 1024，
   和 milvus_utils 的那份会各自漂移），统一调用
   `utils.milvus_utils.ensure_item_name_collection()`，维度取 `get_config().embedding_dim`。
   注意：kb_item_names 的主键字段名仍是 `pk`（kb_chunks 是 `chunk_id`），
   本文件不改字段名，检索侧（node_item_name_confirm / milvus_utils）依赖现状。
5. 别名策略：LLM 可能给出同一商品的多种写法（大小写/空格/全半角/简称）。
   这里做「别名合并」——先归一化+去重，再把与主名同源的别名以
   **别名文本的向量 + 主名的 item_name 字段** 写成额外行：
   检索时命中别名也只会返回主名，而 kb_chunks 的 item_name 过滤值同样是主名，
   因此不会造成「召回的商品名和切片里的商品名对不上」。
"""

import json
import logging
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from config.lm_config import lm_config
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import LLMError, StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import repair_json
from utils.llm_utils import get_llm_client
from utils.milvus_utils import build_in_filter, ensure_item_name_collection

# ==================== 常量 ====================

# kb_item_names 的 file_title / item_name 建表长度是 VARCHAR(100)
# （见 milvus_utils.ensure_item_name_collection），Milvus 按 UTF-8 字节数校验长度，
# 中文 3 字节/字，所以截断必须按字节而不是字符，否则插入直接 DataOverflow
_MILVUS_VARCHAR_BYTES = 100

# 商品名长度上限：超过这个长度基本是「模型把整句话吐回来了」，不是商品名
_ITEM_NAME_MAX_CHARS = 60

# JSON 里可能出现的「主名」键名（中英混写都收，兼容模型自由发挥）
_ITEM_NAME_KEYS = ("item_name", "itemName", "item", "name", "product_name", "productName", "商品名称", "商品名",
                   "产品名称", "产品名", "主体名称", "主体")
# JSON 里可能出现的「别名」键名
_ALIAS_KEYS = ("aliases", "alias", "other_names", "otherNames", "synonyms", "别名", "同义词", "其它写法")

# 模型「没识别出来」时的各种客套写法，统一当成空值走兜底
_PLACEHOLDERS = {"", "无", "未知", "无法识别", "未识别", "没找到", "空", "空字符串", "null", "none", "n/a", "na",
                 "nan", "-", "—", "_", "unknown", "not found", "not applicable"}

# 明显是解释性话术（不是商品名）的关键字
_EXPLANATION_HINTS = ("无法识别", "无法确定", "没有找到", "未找到", "抱歉", "请提供", "建议", "根据文", "根据提")

# 汉字（纯中文商品名的别名重合度校验用）
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# 「商品名称：xxx」这类标签前缀
_LABEL_PREFIX_RE = re.compile(r"^(商品名称|商品名|产品名称|产品名|主体名称|主体|item[\s_-]*names?|name)\s*[:：]\s*",
                              re.IGNORECASE)
# 字符串两端的噪声字符（引号/书名号/Markdown 记号/分隔符），中间的内容不动
_EDGE_NOISE_RE = re.compile(r"^[\s`*_>#\-—–~～\"'“”‘’「」『』《》【】(\[\)）\]]+|"
                            r"[\s`*_>#\-—–~～\"'“”‘’「」『』《》【】(\[\)）\]]+$")
# ```json 代码块围栏（纯文本兜底路径里要先剥掉）
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

# 最多为几个「别名」额外建行：别名脏的时候不至于把商品名集合灌满噪声行
_MAX_ALIAS_ROWS = 2

# 主体识别 prompt：要求 JSON 输出，字段固定，便于解析与后续扩展
_ITEM_NAME_SYSTEM_PROMPT = "你是商品/设备主体识别专家，只负责识别商品名称，严格按要求的 JSON 输出，不输出任何解释。"


# ==================== 纯逻辑函数（无副作用，可单测） ====================

def normalize_item_name(value: Any) -> str:
    """
    商品名归一化：把「同一个商品的不同写法」收敛成一种可比较、可入库的标准形态。
    处理内容：
        1. NFKC：全角字母数字 → 半角（ＨＡＫ１８０ → HAK180）、兼容字符统一
        2. 去首尾噪声（引号/书名号/Markdown 星号/分隔符），剥掉「商品名称：」这类标签前缀
        3. 内部空白折叠成单个空格（保留可读性，也保留 BGE-M3 稀疏向量的分词信息，
           不像课程版那样把所有空格删成 "BrotherHAK180烫金机"）
    :param value: 原始值（允许 None/非字符串）
    :return: 归一化后的字符串；无法归一化时返回空串
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    # Markdown 加粗/行内代码残留（**苏伯尓** 电磁炉 / `HAK180`）
    text = text.replace("**", "").replace("`", "")
    text = _LABEL_PREFIX_RE.sub("", text)
    text = _EDGE_NOISE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 空字符串的多种写法（"N/A"、"—"）归一化后落进占位符表
    if text.casefold().strip(".,;:!?。；：！？") in _PLACEHOLDERS:
        return ""
    return text


def item_name_key(value: Any) -> str:
    """
    商品名「比较键」：在 normalize_item_name 之上再抹掉大小写/空格/连接符差异。
    用途：去重（"HAK180 烫金机" 与 "hak-180烫金机" 视为同一个）。
    :return: 归一化、casefold、删除空白与常见分隔符后的键；空值返回空串
    """
    name = normalize_item_name(value)
    if not name:
        return ""
    return re.sub(r"[\s\-_/·•.,，、;；:：()（）\[\]]+", "", name).casefold()


def _is_valid_item_name(name: str) -> bool:
    """候选名是否是「像商品名」的有效值：非空、非占位符、不像整句解释、长度合理。"""
    if not name or name in _PLACEHOLDERS:
        return False
    if len(name) > _ITEM_NAME_MAX_CHARS:
        return False
    # 残留的 JSON/结构字符：说明这行是模型的话术或半截 JSON，不是商品名
    if re.search(r"[{}\[\]<>\"']", name):
        return False
    # 「无法识别商品名称」这类话术里含动词/标点，直接判掉
    return not any(hint in name for hint in _EXPLANATION_HINTS)


def dedupe_item_names(values: Iterable[Any]) -> List[str]:
    """
    商品名列表去重（保序）：逐个归一化 → 过滤无效值 → 按比较键去重。
    :param values: 任意来源的名称（可能含 None、重复写法、解释话术）
    :return: 归一化且去重后的名称列表（最多一个同义写法被保留）
    """
    seen: set[str] = set()
    result: List[str] = []
    for value in values or []:
        name = normalize_item_name(value)
        if not _is_valid_item_name(name):
            continue
        key = item_name_key(name)
        if key in seen:
            continue
        seen.add(key)
        result.append(name)
    return result


def pick_primary_item_name(names: Sequence[str]) -> str:
    """
    从候选名里挑「主名」：商品名越完整越好检索（品牌 + 型号 + 品名）。
    打分优先级：含型号数字 > 含拉丁字母（品牌） > 长度（更完整）。
    例：["烫金机", "Brother HAK180 烫金机"] → 选后者。
    :param names: 已归一化/去重的候选名
    :return: 主名；无有效候选返回空串
    """
    if not names:
        return ""

    def score(name: str) -> Tuple[int, int, int]:
        folded = name.casefold()
        has_digit = 1 if re.search(r"\d", folded) else 0
        has_latin = 1 if re.search(r"[a-z]", folded) else 0
        return has_digit, has_latin, len(name)

    return max(names, key=score)


def _alnum_parts(name: str) -> Tuple[set, set]:
    """
    把一个名字拆成「字母段」和「数字段」，用来判断两个写法是不是同一个商品。
    为什么不用整段 alnum 比对：去掉空格/连字符后 "Brother HAK180" 会粘成
    "brotherhak180"，而别名 "HAK-180" 是 "hak180"，整段永远不相等；
    拆开后数字段都是 {180}，才是真正可比的型号信号。
    :return: (字母段集合, 数字段集合)
    """
    key = item_name_key(name)
    return set(re.findall(r"[a-z]+", key)), set(re.findall(r"\d+", key))


def _alias_is_same_product(alias: str, primary: str) -> bool:
    """
    别名与主名是否同源（同一个商品的不同写法）。
    判定顺序（宁可漏，不可错：错收录的别名会把别的商品的问题引到本商品上）：
        1. 两边都带型号数字时必须相同，否则一定是两台设备（5000W vs 3000W）
        2. 数字段或字母段有交集 → 同源
        3. 都没有可比对的字母/数字（纯中文品名）→ 汉字重合度 ≥ 50%
    """
    primary_alpha, primary_digit = _alnum_parts(primary)
    alias_alpha, alias_digit = _alnum_parts(alias)
    if primary_digit and alias_digit and not (primary_digit & alias_digit):
        return False
    if primary_digit & alias_digit or primary_alpha & alias_alpha:
        return True
    primary_cjk = set(_CJK_RE.findall(primary))
    if not primary_cjk:
        return False
    return len(primary_cjk & set(_CJK_RE.findall(alias))) / len(primary_cjk) >= 0.5


def filter_related_aliases(aliases: Sequence[str], primary: str, limit: int = _MAX_ALIAS_ROWS) -> List[str]:
    """
    别名筛选：只保留「确实指向同一个商品」的写法，防止模型把别的商品/泛化词
    （如「产品」「手册」）混进别名，反而污染商品名索引。
    规则：与主名去重（比较键不同才保留）+ _alias_is_same_product 同源校验 + 条数上限。
    :param aliases: 归一化去重后的别名候选（不含主名）
    :param primary: 主名
    :param limit: 最多保留几条
    :return: 可安全入库的别名列表
    """
    primary = normalize_item_name(primary)
    if not primary:
        return []
    primary_key = item_name_key(primary)

    kept: List[str] = []
    for alias in aliases:
        alias = normalize_item_name(alias)
        if not _is_valid_item_name(alias):
            continue
        alias_key = item_name_key(alias)
        if not alias_key or alias_key == primary_key:
            continue
        if not _alias_is_same_product(alias, primary):
            continue
        if any(alias_key == item_name_key(k) for k in kept):
            continue
        kept.append(alias)
        if len(kept) >= limit:
            break
    return kept


def build_item_name_prompt(file_title: str, context: str) -> str:
    """
    组装主体识别的用户 prompt（抽成纯函数便于单测与调参）。
    明确要求 JSON 结构，同时保留课程规格里的「品牌+型号+品名」示例与空值约定。
    """
    return f"""请从以下信息中识别出这份文档所属的商品名称与型号。

文件名：{file_title}

正文切片（用于辅助识别）：
{context}

输出要求：
1. 只输出一个 JSON 对象，不要输出任何解释文字，也不要使用 ``` 代码块包裹；
2. JSON 结构固定为：{{"item_name": "完整商品名称", "aliases": ["同一商品的其他写法"]}}；
3. item_name 最好是带品牌、型号和名称的完整商品名称，例如：苏伯尓5000W大功率电磁炉；
4. aliases 只收录「同一个商品」的其他写法（大小写、空格、全半角、中英文、简称），
   没有就返回空数组；不同商品、泛化词（如「产品」「手册」）一律不要放进来；
5. 如果无法识别商品名称，item_name 返回空字符串，aliases 返回空数组。
"""


def parse_item_name_result(raw: Any) -> Tuple[str, List[str]]:
    """
    把大模型返回的原始文本解析成 (主名, 别名列表)。

    三级容错（逐级降级，任何一级拿到有效值就返回）：
        1. repair_json：剥 ```json 围栏 / 补尾逗号 / 单引号字面量 / 截断闭合，
           拿到 dict 后按多种键名取值（item_name / name / 商品名 ...）
        2. JSON 是数组（模型把对象写成列表）→ 取元素（元素可为 dict 或 str）
        3. 完全不是 JSON（小模型直接吐一行商品名，或一句「无法识别」）
           → 纯文本兜底：剥围栏、按行取第一条像商品名的文本
    别名在解析阶段只做归一化+去重，「是否同源」交给 filter_related_aliases 判断。

    :param raw: LLM 原始输出（str；也兼容已经是 dict/list 的情况）
    :return: (主名, 别名列表)；识别失败时主名为空串，由调用方决定兜底值
    """
    if raw is None:
        return "", []

    # expect 传 (dict, list)：repair_json 内部用 isinstance(obj, expect)，支持元组
    data = repair_json(raw, default=None, expect=(dict, list))

    if isinstance(data, dict):
        primary_raw = _first_present(data, _ITEM_NAME_KEYS)
        aliases_raw = _first_present(data, _ALIAS_KEYS)
        names = dedupe_item_names(_flatten_names([primary_raw]))
        aliases = dedupe_item_names(_flatten_names([aliases_raw]))
        if not names:
            # 键名对不上（模型自己编了个键）：把 dict 的所有值当候选名试一次
            names = dedupe_item_names(_flatten_names(list(data.values())))
        primary = pick_primary_item_name(names)
        if primary:
            aliases = [n for n in aliases if item_name_key(n) != item_name_key(primary)]
            return primary, aliases
        # dict 里什么有效名都没有 → 继续走纯文本兜底

    if isinstance(data, list):
        names = dedupe_item_names(_flatten_names(data))
        if names:
            primary = pick_primary_item_name(names)
            return primary, [n for n in names if item_name_key(n) != item_name_key(primary)]

    # 兜底：当作纯文本处理
    names = dedupe_item_names(_plain_text_candidates(raw))
    if not names:
        return "", []
    primary = pick_primary_item_name(names)
    return primary, [n for n in names if item_name_key(n) != item_name_key(primary)]


def _first_present(data: Dict[str, Any], keys: Sequence[str]) -> Any:
    """按候选键名顺序取第一个非空值（模型输出键名不稳定）。"""
    for key in keys:
        if key in data and data[key] not in (None, "", [], {}):
            return data[key]
    return None


def _flatten_names(values: Iterable[Any]) -> List[Any]:
    """
    把「字符串 / 列表 / 字典列表」三种形态压平成字符串候选列表。
    模型经常把 item_name 写成 ["A", "B"] 或 [{"item_name": "A"}]，都要能吃下。
    """
    flat: List[Any] = []
    for value in values or []:
        if value is None:
            continue
        if isinstance(value, str):
            # 一行一个写法的情况（"A\nB"）也拆开
            flat.extend(line for line in value.splitlines() if line.strip())
        elif isinstance(value, dict):
            flat.extend(_flatten_names([_first_present(value, _ITEM_NAME_KEYS), _first_present(value, _ALIAS_KEYS)]))
        elif isinstance(value, (list, tuple, set)):
            flat.extend(_flatten_names(value))
        else:
            flat.append(value)
    return flat


def _plain_text_candidates(raw: Any) -> List[str]:
    """
    非 JSON 输出的兜底解析：剥掉 ``` 围栏，逐行找第一条「像商品名」的文本。
    为什么要限长并排除标点：模型的解释句（"根据文件内容，无法识别商品名称。"）
    必须在这一层被拒掉，否则会把整句话当商品名写进库。
    """
    text = str(raw or "")
    match = _FENCE_RE.search(text)
    text = match.group(1) if match else text

    candidates: List[str] = []
    for line in text.splitlines():
        name = normalize_item_name(line)
        if not name:
            continue
        if not _is_valid_item_name(name):
            continue
        # 含句末/分句标点的多半是一句话而不是商品名（型号里的连字符不算）
        if re.search(r"[。！？；]", name):
            continue
        candidates.append(name)
    return candidates


def build_item_name_context(chunks: Sequence[Dict], chunk_k: int, total_size: int) -> str:
    """
    构造主体识别的上下文（步骤 2 的纯实现）。
    与课程版的差异（都是踩过的坑）：
        1. 取块：不是死取前 k 个，而是「跳过空切片后凑够 k 个有效切片」，
           避免图片占位/空标题切片白占上下文预算；
        2. 预算：单片配额 = total_size // k，逐片截断，
           而不是拼接完再整体砍到 total_size —— 后者只会保留第 1 片，
           等于白取 k 片；
        3. 健壮：chunk 缺 title/content（上游节点字段不齐）时 .get(...).strip()
           会抛 AttributeError，这里统一按空串处理。
    :param chunks: 文本切片列表（[{title, content, ...}]）
    :param chunk_k: 最多取几个有效切片（config.item_name_chunk_k，本机 3）
    :param total_size: 上下文字符总上限（config.item_name_chunk_size，本机 2500）
    :return: 形如「【切片1】\n标题：xxx\n内容：yyy」的结构化上下文；无有效内容返回空串
    """
    k = max(1, int(chunk_k or 1))
    total_size = max(0, int(total_size or 0))
    if total_size <= 0 or not chunks:
        return ""
    # 每片正文配额：留一点空间给「【切片N】标题：」这些包装文字
    per_chunk_budget = max(100, total_size // k - 50)

    parts: List[str] = []
    for chunk in chunks:
        if len(parts) >= k:
            break
        if not isinstance(chunk, dict):
            continue
        title = normalize_item_name(chunk.get("title")) or "无标题"
        content = re.sub(r"\s+", " ", str(chunk.get("content") or "")).strip()
        if not content:
            continue
        if len(content) > per_chunk_budget:
            content = content[:per_chunk_budget]
        parts.append(f"【切片{len(parts) + 1}】\n标题：{title}\n内容：{content}")

    context = "\n\n".join(parts).strip()
    # 二次硬截断：包装文字叠起来仍可能略超上限，最后再兜一层保证不超模型输入限制
    return context[:total_size]


def clamp_milvus_varchar(value: Any, max_bytes: int = _MILVUS_VARCHAR_BYTES) -> str:
    """
    按 UTF-8 字节数截断到 Milvus VARCHAR 的 max_length 以内。
    为什么按字节：Milvus 校验的是字节长度，中文 3 字节/字，
    按 100 字符截断的中文名（33 字以上）会在 insert 时直接报长度超限。
    decode 用 errors="ignore"，避免把汉字切成半个字节序列。
    """
    if value is None:
        return ""
    text = str(value)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def build_item_name_rows(file_title: str, primary_item_name: str,
                         vectors: Sequence[Tuple[Any, Any]]) -> List[Dict[str, Any]]:
    """
    组装 kb_item_names 的待插入行（步骤 6 的纯实现）。
    约定：vectors[i] 对应「第 i 个用于向量化的文本」，第 0 个是主名，其后是别名；
    但**所有行的 item_name 字段一律写主名** —— 检索端 node_item_name_confirm
    取 output_fields=["item_name"] 后拿它去过滤 kb_chunks，若别名行写别名，
    过滤值就会和切片里的 item_name 对不上，导致召回为空。
    :param file_title: 源文件标题（幂等删除依据）
    :param primary_item_name: 归一化后的主名
    :param vectors: [(dense_vector, sparse_vector), ...]，长度与向量化文本数一致
    :return: 符合集合 Schema 的行列表；缺向量的行跳过（FLOAT_VECTOR 非空，写了必失败）
    """
    item_name = clamp_milvus_varchar(primary_item_name)
    safe_title = clamp_milvus_varchar(file_title)
    rows: List[Dict[str, Any]] = []
    for dense_vector, sparse_vector in vectors or []:
        if not dense_vector or not sparse_vector:
            continue
        rows.append({
            "file_title": safe_title,
            "item_name": item_name,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        })
    return rows


def extract_llm_text(response: Any) -> str:
    """
    从 LLM 响应对象里取出正文文本，兼容各家客户端的差异。
    取值顺序：content（str / 内容块列表）→ additional_kwargs 的
    reasoning_content / thinking → response_metadata 同名键。
    :raises LLMError: 所有字段都没有内容时抛出（宁可不写，也不把空结果当识别结果）
    """
    if response is None:
        raise LLMError(message="大模型返回为空对象，无法提取商品名")
    if isinstance(response, str):
        text = response
    else:
        text = _stringify_content(getattr(response, "content", None))
        if not text:
            for holder in (getattr(response, "additional_kwargs", None), getattr(response, "response_metadata", None)):
                if not isinstance(holder, dict):
                    continue
                for key in ("reasoning_content", "thinking", "content"):
                    text = _stringify_content(holder.get(key))
                    if text:
                        break
                if text:
                    break
    if not text.strip():
        raise LLMError(message="大模型返回内容为空（content/reasoning_content 均无正文），拒绝写入空商品名")
    return text


def _stringify_content(content: Any) -> str:
    """content 可能是 str，也可能是 langchain 1.x 的内容块列表 [{'type':'text','text':...}]。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, (list, tuple)):
        return "".join(_stringify_content(item) for item in content)
    return str(content)


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph 核心节点：商品主体名称识别
        流程总览：
            1. 提取输入
            2. 构建大模型上下文
            3. 调用大模型识别商品名称（JSON + 脏输出容错 + 归一化去重）
            4. 回填商品名称到状态和切片
            5. 生成商品名称（含别名）的稠密/稀疏向量
            6. 将数据存入Milvus向量数据库

        必要参数：task_id、file_title, chunks
        更新参数：item_name、chunks（每个切片补 item_name）

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：提取并校验输入
        file_title, chunks = self._step_1_get_inputs(state)

        # 步骤2：构建大模型识别的上下文（前 k 个有效切片，总长受 item_name_chunk_size 约束）
        context = self._step_2_build_context(chunks)

        # 步骤3：调用大模型识别商品名称（主名 + 同源别名）
        item_name, aliases = self._step_3_call_llm(file_title, context)

        # 步骤4：回填商品名称到状态和切片（必须在写库之前，保证 state/chunks/集合三处一致）
        self._step_4_update_chunks(state, chunks, item_name)

        # 步骤5：为「主名 + 别名」批量生成稠密/稀疏向量（一次调用，省 CPU 推理开销）
        vectors = self._step_5_generate_vectors([item_name, *aliases])

        # 步骤6：将数据存入Milvus向量数据库
        self._step_6_save_to_milvus(state, file_title, item_name, vectors)

        # 打印识别结果
        self.logger.info(f"--- 识别完成: {item_name} (别名: {aliases or '无'}) ---")

        return state

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, List[Dict]]:
        """
        步骤 1: 接收并校验流程输入
        核心作用：
            1. 从流程状态中提取文件标题、文本切片核心数据
            2. 基础数据类型校验，保证下游流程输入有效性
        依赖的状态数据（上游节点产出）：
            - state["file_title"]: 上游提取的文件标题
            - state["chunks"]: 文本切片列表
        返回值：
            Tuple[str, List[Dict]]: (处理后的文件标题, 校验后的文本切片列表)
        """

        # 1、参数校验    鲁棒性（健壮性） 校验
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        # file_title 上游可能带扩展名/首尾空格，先归一化，保证 state、切片、Milvus 三处同值
        return str(file_title).strip(), chunks

    def _step_2_build_context(self, chunks: List[Dict]) -> str:
        """
        步骤 2: 构造大模型商品名称识别的标准化上下文
        取块/截断细节见 build_item_name_context（纯函数，便于单测）。
        参数说明：
            chunks: 文本切片列表
        返回值：
            str: 格式化后的上下文字符串（无有效内容时为空串，由步骤3走标题兜底）
        """
        context = build_item_name_context(
            chunks,
            chunk_k=self.config.item_name_chunk_k,
            total_size=self.config.item_name_chunk_size,
        )
        if not context:
            self.logger.warning("步骤2：未能从切片中构造出有效上下文，将直接用文件标题兜底识别结果")
        return context

    def _step_3_call_llm(self, file_title: str, context: str) -> Tuple[str, List[str]]:
        """
        步骤 3: 调用大模型实现商品名称/型号精准识别
        核心逻辑：
            1. 上下文为空 → 直接返回 file_title（兜底，无需调用大模型）
            2. 上下文非空 → 用便宜档 item_model(deepseek-flash) 按 JSON prompt 抽取
            3. 返回值走 parse_item_name_result 做脏 JSON 容错 + 归一化 + 去重 + 别名合并
            4. 识别不到（含内容为空的硬性失败、异常）→ 用 file_title 兜底，保证流程不中断，
               且**永不返回空商品名**（空商品名会写进 Milvus/state，污染下游检索）
        参数：
            file_title: 处理后的文件标题（异常/空值时的兜底值）
            context: 步骤2构建的结构化切片上下文（大模型识别的核心依据）
        返回值：
            Tuple[str, List[str]]: (标准商品名, 同源别名列表)
        """

        # 兜底名同样要归一化并满足 VARCHAR 长度约束（state 与 Milvus 存同一个值）
        fallback = self._fallback_item_name(file_title)

        # 1. 上下文为空，返回file_title
        if not context:
            return fallback, []

        try:
            # 2.1 组装 prompt（要求 JSON 输出）
            prompt = build_item_name_prompt(file_title=file_title, context=context)

            # 2.2 初始化大模型客户端（实体抽取走便宜档 item_model；get_llm_client 内部按模型缓存单例）
            llm = get_llm_client(model=self._item_model())

            # 2.3 创建消息对象：SystemMessage 定人设 + HumanMessage 传业务请求
            chat_messages = [
                SystemMessage(content=_ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            # 2.4 模型调用 + 2.5 容错取值（content/reasoning_content 全空会抛 LLMError）
            raw_text = extract_llm_text(llm.invoke(chat_messages))
            self.logger.info(f"步骤3：大模型原始返回：{raw_text[:200]}")

            # 2.6 解析 + 归一化 + 去重 + 别名同源校验
            primary, aliases = parse_item_name_result(raw_text)
            if not primary:
                self.logger.warning(f"步骤3：未能从大模型返回中识别出商品名，使用文件标题兜底：{fallback}")
                return fallback, []
            aliases = filter_related_aliases(aliases, primary)

            # 2.7 统一按 Milvus VARCHAR(100 字节) 截断，保证「state / 切片 / 集合」三处同值
            clamped_primary = clamp_milvus_varchar(primary)
            if clamped_primary != primary:
                self.logger.warning(f"步骤3：商品名超出 Milvus VARCHAR 限制，已截断：{primary} -> {clamped_primary}")
            clamped_aliases = [clamp_milvus_varchar(alias) for alias in aliases]
            # 截断后可能撞车（两个长名字截断成同一个），再去一次重，避免写入完全重复的行
            clamped_aliases = [a for a in clamped_aliases if item_name_key(a) != item_name_key(clamped_primary)]
            return clamped_primary, clamped_aliases

        except LLMError as e:
            # 内容为空属于「大模型不可用/异常」级别的错误，必须显式记录（不静默写空结果）
            self.logger.error(f"步骤3：大模型返回不可用，使用文件标题兜底，原因：{e}")
            return fallback, []
        except Exception:  # 兜底策略：任何 LLM 侧异常都不能中断导入，统一用文件标题兜底
            # 网络超时、鉴权失败、SDK 内部异常等一律兜底，exception() 自带堆栈不必再拼 {e}
            self.logger.exception("步骤3：大模型调用失败，使用文件标题兜底")
            return fallback, []

    def _fallback_item_name(self, file_title: str) -> str:
        """
        兜底商品名：归一化文件标题；标题本身被归一化空了（如标题就叫「无」）
        则退回原始标题字符串，最后保证长度满足 VARCHAR 约束。
        """
        name = normalize_item_name(file_title) or str(file_title or "").strip()
        return clamp_milvus_varchar(name or "未命名商品")

    @staticmethod
    def _item_model() -> str:
        """
        主体识别使用的模型名：优先 lm_config.item_model（本机 = deepseek-flash）。
        配置缺失时退回 llm_model，避免 KeyError 让整个节点崩掉。
        """
        return lm_config.item_model or lm_config.llm_model

    def _step_4_update_chunks(self, state: ImportGraphState, chunks: List[Dict[str, str]], item_name: str):
        """
        步骤 4: 回填商品名称到流程状态和所有文本切片
        核心作用：
            1. 全局状态更新：将item_name存入state，供下游所有节点直接使用
            2. 切片数据补全：为每个切片添加item_name字段，保证数据一致性
            3. 状态同步：更新state中的chunks，确保切片修改全局生效
        设计思路：
            所有切片关联同一商品名称，且与 kb_item_names 写入的是同一个字符串，
            检索端才能用「库里的商品名」过滤到「切片里的商品名」。
        参数：
            state: 流程状态对象
            chunks: 校验后的文本切片列表
            item_name: 步骤3识别并清洗后的商品名称
        """

        state["item_name"] = item_name

        # 遍历所有的片段
        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk["item_name"] = item_name

        state["chunks"] = chunks

    def _step_5_generate_vectors(self, names: List[str]) -> List[Tuple[Any, Any]]:
        """
        步骤 5: 为商品名称（含别名）生成 BGE-M3 稠密 + 稀疏双向量
        核心说明：
            - 稠密向量：维度取配置 embedding_dim（本机 BGE-M3 = 1024），不写死
            - 稀疏向量：{token_id: 权重}，key 必须是原生 int（embedding_utils 已转换）
            - 主名和别名一次批量调用，减少 CPU 上的模型前向次数
        健壮性：
            维度与配置不符 / 返回结构长度不一致 → 视为无效向量（返回空槽），
            由步骤 6 跳过该行，避免把注定插入失败的向量塞进 Milvus。
        参数：
            names: 需要向量化的文本（[主名, *别名]）
        返回值：
            List[Tuple[Any, Any]]: 与 names 等长的 [(dense, sparse), ...]，无效为 (None, None)
        """
        names = [name for name in (names or []) if name]
        if not names:
            return []

        try:
            vectors = generate_embeddings(names)
            dense_list = vectors.get("dense") or []
            sparse_list = vectors.get("sparse") or []
        except Exception:  # 向量化失败只影响商品名集合，不该中断导入主流程（切片入库会再次暴露该问题）
            self.logger.exception("步骤5：商品名向量化失败，跳过商品名集合写入")
            return []

        expected_dim = int(self.config.embedding_dim)
        results: List[Tuple[Any, Any]] = []
        for idx in range(len(names)):
            dense = dense_list[idx] if idx < len(dense_list) else None
            sparse = sparse_list[idx] if idx < len(sparse_list) else None
            if not dense or not sparse:
                self.logger.error(f"步骤5：第{idx + 1}个商品名「{names[idx]}」的向量生成不完整，跳过该行")
                results.append((None, None))
                continue
            if len(dense) != expected_dim:
                self.logger.error(
                    f"步骤5：稠密向量维度 {len(dense)} 与配置 embedding_dim={expected_dim} 不符，"
                    f"请检查 EMBEDDING_DIM 与模型是否匹配，跳过该行")
                results.append((None, None))
                continue
            results.append((dense, sparse))
        return results

    def _step_6_save_to_milvus(self, state: ImportGraphState, file_title: str, item_name: str,
                               vectors: List[Tuple[Any, Any]]):
        """
        步骤 6: 将商品名称、文件标题、双向量持久化到 Milvus 向量数据库
        核心逻辑：
            1. 配置校验：集合名缺失直接跳过（环境没配好不该报错打断导入）
            2. 建集合：统一走 milvus_utils.ensure_item_name_collection
               （幂等创建 + 维度取 EMBEDDING_DIM，本文件不再重复一份 Schema）
            3. 幂等清理：按 file_title 或 item_name 删除旧行（重跑同名文档不产生重复数据）
            4. 数据插入：批量插入，行内容符合集合 Schema
        降级策略：
            Milvus 任何异常/向量缺失 → 只记日志不抛穿，因为切片入库节点（node_import_milvus）
            是主数据；商品名集合缺失只影响「检索时的名字对齐」，可重跑补齐。
        参数：
            state: 流程状态对象，用于最终状态同步
            file_title: 处理后的文件标题
            item_name: 识别后的商品名称（幂等去重依据）
            vectors: 步骤 5 生成的 [(dense, sparse), ...]，第 0 个对应主名
        """

        collection_name = milvus_config.item_name_collection
        if not collection_name:
            self.logger.warning("跳过 Milvus 保存：缺少 ITEM_NAME_COLLECTION 配置")
            return

        rows = build_item_name_rows(file_title=file_title, primary_item_name=item_name, vectors=vectors)
        if not rows:
            self.logger.error(f"步骤6：商品名「{item_name}」没有可用的双向量，跳过 kb_item_names 写入")
            return

        try:
            # 1. 获取/创建集合（ensure_* 内部已处理 has_collection、索引与 load_collection）
            milvus_client = ensure_item_name_collection(vector_dimension=int(self.config.embedding_dim))

            # 2. 幂等处理：删除「同一文件的旧记录」和「同名商品的旧记录」
            #    用 build_in_filter 生成双引号字面量并正确转义，避免手拼引号被商品名里的引号打爆
            exprs = [
                expr for expr in (
                    build_in_filter("file_title", [clamp_milvus_varchar(file_title)]),
                    build_in_filter("item_name", [clamp_milvus_varchar(item_name)]),
                ) if expr
            ]
            if exprs:
                milvus_client.delete(collection_name=collection_name, filter=" or ".join(exprs))

            # 3. 批量插入（主名行 + 别名行；item_name 字段全部是主名）
            milvus_client.insert(collection_name=collection_name, data=rows)
            self.logger.info(f"将商品名称：'{item_name}' 共 {len(rows)} 行写入 Milvus 集合 {collection_name}")

            # 4. 状态确认：state 已在步骤4回填，这里只做一致性兜底（防止外部直接调用本步骤）
            state["item_name"] = clamp_milvus_varchar(item_name)

        except Exception as e:  # noqa: BLE001  # 降级处理：Milvus 任何异常都不该中断导入主流程
            # 切片入库由 node_import_milvus 负责，商品名集合缺失可重跑补齐
            self.logger.warning(f"数据存入Milvus失败，跳过商品名保存，错误原因：{e}")


if __name__ == "__main__":
    # 手工调试入口：需要真实 DeepSeek + Milvus 在线（静态修复阶段不要跑这里）
    setup_logging()

    json_path = r"D:\output\hak180产品安全手册\chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        chunks_json = f.read()

    init_state = {
        "task_id": "local-item-name-test",
        "chunks": json.loads(chunks_json),
        "file_title": "hak180产品安全手册",
    }

    # 执行核心处理流程
    node_item_name_recognition = NodeItemNameRecognition()
    result = node_item_name_recognition(init_state)

    state_json_path = r"D:\output\hak180产品安全手册\state.json"
    with open(state_json_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False 保证中文字符不变成 \uXXXX 编码，indent=4 让输出格式化美观
        json.dump(result, f, ensure_ascii=False, indent=4)

    logging.getLogger().info(f"状态回填成功！最终 state 已安全写入文件: {state_json_path}")
