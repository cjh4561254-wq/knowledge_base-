# processor/query_processor/nodes/node_item_name_confirm.py

"""检索流程第一个节点：商品名确认。

流程：向量召回商品名候选（Milvus kb_item_names）→ LLM 提取/判定商品名并改写问题
→ 评分对齐（确认 / 候选 / 无结果）→ 无法确认时把追问话术写入 state["answer"]，
由 main_graph 的条件路由直接跳到 node_answer_output 返回给用户（交互闭环见报告说明）。

本文件的健壮性约定（为什么这样写）：
1. MongoDB 只是"副作用"存储（聊天历史）：任何一次读写失败都降级为 WARNING 并继续，
   绝不允许因为历史写入失败把整条查询图打挂。
2. LLM 返回不可信：内容可能被 ```json 围栏包裹、带解释性前置文本、尾逗号、单引号、
   被 max_tokens 截断，甚至 content 为空（推理模型把正文全放进 reasoning_content）。
   统一走 utils.json_format_utils.repair_item_names_result 修复解析；
   全空/解析彻底失败时兜底为「原始问题原样传下去 + 空 item_names」，
   绝不写入空 rewritten_query（那会让后续检索全废）。
3. 大模型档位：只使用 config.lm_config.lm_config.item_model（本机 = deepseek-flash），
   账号没有 deepseek-chat / deepseek-reasoner，硬编码其它模型名会直接 400。
4. 「LLM 返回 → 解析 → 候选过滤 → 对齐」抽成模块级纯函数（无 IO），
   可在不连 Milvus / 不调 DeepSeek 的情况下单测。
"""

from typing import Any, Dict, List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config.lm_config import lm_config
from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import repair_item_names_result
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from utils.mongo_history_utils import save_chat_message, get_recent_messages, update_message_item_names
from utils.task_utils import add_done_task

# 对齐阈值（课程 day12 规格：>0.85 确信、0.6~0.85 候选、<0.6 丢弃）
HIGH_SCORE_THRESHOLD = 0.85
MID_SCORE_THRESHOLD = 0.6
# 单个提取名最多召回的候选商品数（与 step5 的 limit 一致）
CANDIDATE_TOP_N = 5
# 分支 B 追问话术中最多展示的候选数（避免提示语过长）
OPTIONS_DISPLAY_N = 3


# ======================================================================
# 纯函数层：LLM 响应取值 / JSON 修复 / 候选去重 / 评分对齐 / 分支判定
# （无网络、无数据库副作用，便于临时单测覆盖脏 JSON、空 content、别名行）
# ======================================================================

def extract_llm_text(response: Any) -> str:
    """
    从 ChatOpenAI 响应对象里取出可解析的文本。
    为什么要容错：推理型模型可能把 content 留空、正文写进 reasoning_content；
    langchain 1.x 的 content 还可能是分块列表（[{type:text,...}]）。
    取值顺序：content(字符串/分块列表) → reasoning_content(属性/additional_kwargs)，
    全空则抛 ValueError（调用方兜底，绝不带着空文本继续解析）。
    """
    def _flatten(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            # 分块列表：兼容 {"text": ...} / {"content": ...} / 裸字符串 三种形态
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(str(block.get("text") or block.get("content") or ""))
            return "".join(parts).strip()
        return ""

    text = _flatten(getattr(response, "content", None))
    if text:
        return text

    # content 为空 → 试 reasoning_content（DeepSeek 推理模式 / 部分网关会这样返回）
    text = _flatten(getattr(response, "reasoning_content", None))
    if not text:
        extra = getattr(response, "additional_kwargs", None) or {}
        text = _flatten(extra.get("reasoning_content"))
    if text:
        logger.warning("LLM 返回 content 为空，已降级使用 reasoning_content 作为正文")
        return text

    raise ValueError("大模型返回内容为空（content 与 reasoning_content 均为空）")


def normalize_item_names(names: Any) -> List[str]:
    """
    清洗商品名列表：去掉每项内部空白（课程规格：HAK 180 → HAK180，对齐库内标准名）、
    过滤空串、保持出现顺序去重。
    """
    if not isinstance(names, (list, tuple)):
        return []
    cleaned: List[str] = []
    seen = set()
    for name in names:
        s = str(name).replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def parse_item_names_result(raw_text: str, original_query: str) -> Dict[str, Any]:
    """
    把 LLM 原始输出修复解析为节点契约结构：
        {"item_names": List[str], "rewritten_query": str}
    为什么复用 repair_item_names_result：项目 json_format_utils 已实现
    围栏剥离/前置文本丢弃/尾逗号/单引号/截断补全的逐级修复，
    这里只需再补两件事——商品名去内部空白、rewritten_query 兜底为非空
    （空改写会让下游向量检索全部跑偏，必须回落原始问题）。
    """
    result = repair_item_names_result(raw_text, original_query=original_query)
    result["item_names"] = normalize_item_names(result.get("item_names"))
    rewritten = (result.get("rewritten_query") or "").strip()
    result["rewritten_query"] = rewritten or original_query
    return result


def dedupe_matches_by_item_name(matches: List[Dict]) -> List[Dict]:
    """
    按 item_name 去重匹配结果，同名保留最高分。
    为什么必须做：导入侧 kb_item_names 含"别名行"——别名的 item_name 字段写的也是主名，
    同一主名会以多条向量记录的身份被召回；不去重会把候选列表占满、
    还会把"多条 >0.85"误判成需要用户确认（实际是同一商品）。
    """
    best: Dict[str, float] = {}
    order: List[str] = []
    for m in matches or []:
        name = (m.get("item_name") or "").strip()
        if not name:
            continue
        score = m.get("score") or 0.0
        if name not in best:
            best[name] = score
            order.append(name)
        else:
            best[name] = max(best[name], score)
    return [{"item_name": name, "score": best[name]} for name in order]


def dedupe_preserve(matches: List[str]) -> List[str]:
    """
    保序去重 + 只去首尾空白，**保留库内原始写法**（与 normalize_item_names 的关键区别）。

    为什么必须新开这个函数（真实链路回归修复）：
    normalize_item_names 会删掉词内所有空格，那是给「LLM 提取出的原始文本」用的
    （HAK 180 → HAK180，让它去对齐库内标准名）。但本函数处理的是
    `align_item_names` 里从 Milvus `kb_item_names` 的 entity 取回的**库内标准名**，
    这些名字会被写进 state["item_names"]，最终由 node_search_embedding 拼成
    `item_name in ["..."]` 过滤表达式 —— Milvus varchar 的 in 是逐字符精确匹配，
    多一个空格就 0 命中。
    实测本机：库内名是 "Coze 工作流实战:商家客服初级"（Coze 后有空格），
    被 normalize 成 "Coze工作流实战:商家客服初级" 后
    → 向量路与 HyDE 路双双「候选 0 条」，本地召回被整体抹掉，答案只能靠网络搜索。
    所以库内名必须原样透传。

    :param matches: 名称列表（可能来自库内，也可能含 None / 空串）
    :return: 去空白（仅首尾）、去重、保序的列表
    """
    cleaned: List[str] = []
    seen = set()
    for name in matches:
        if name is None:
            continue
        s = str(name).strip()
        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def align_item_names(
    query_results: List[Dict],
    high_threshold: float = HIGH_SCORE_THRESHOLD,
    mid_threshold: float = MID_SCORE_THRESHOLD,
    top_n: int = CANDIDATE_TOP_N,
) -> Dict[str, List[str]]:
    """
    评分对齐（纯函数，规则 a>b>c>d，与课程 day12 一致）：
        a 恰有一条 >high → 直接确认
        b 多条 >high → 优先取与提取名完全一致的，否则取第一条（分数最高）
        c 无 >high → ≥mid 的前 top_n 条作为候选
        d 全部 <mid → 确认与候选均为空
    与课程版的差异（修复点）：
    - 先对 matches 做别名行去重（保留最高分），再做阈值判定；
    - 去重从 list(set()) 改为保序去重，前端候选展示不再随机漂移。
    """
    confirmed_item_names: List[str] = []
    options: List[str] = []

    for res in query_results or []:
        extracted_name = (res.get("extracted_name", "") or "").strip()
        # 别名行去重：同名多条召回只留最高分
        matches = dedupe_matches_by_item_name(res.get("matches", []) or [])
        if not matches:
            continue

        # 按分数降序排列，保证"分数最高的第一条"语义成立
        matches.sort(key=lambda m: m.get("score", 0), reverse=True)
        high = [m for m in matches if (m.get("score") or 0) > high_threshold]
        mid = [m for m in matches if (m.get("score") or 0) >= mid_threshold]

        # 规则a：唯一高置信度 → 直接确认
        if len(high) == 1:
            confirmed_item_names.append(high[0].get("item_name"))
            continue

        # 规则b：多条高置信度 → 优先与原始提取名相同，否则取最高分
        if len(high) > 1:
            picked = next((m for m in high if extracted_name and m.get("item_name") == extracted_name), None)
            confirmed_item_names.append((picked or high[0]).get("item_name"))
            continue

        # 规则c：无高置信度 → ≥mid 的前 top_n 作为候选
        # 规则d：mid 也为空时自然不追加任何内容
        for m in mid[:top_n]:
            options.append(m.get("item_name"))

    return {
        # 库内标准名原样透传（见 dedupe_preserve 的说明）：这里若再用
        # normalize_item_names 剥掉词内空格，下游 item_name in [...] 精确过滤会 0 命中
        "confirmed_item_names": dedupe_preserve(confirmed_item_names),
        "options": dedupe_preserve(options),
    }


def resolve_confirmation(align_result: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    分支判定（纯函数）：把对齐结果映射为 (最终 item_names, 追问/拒识 answer 话术)。
    返回值语义（与 state 字段契约一致，交互闭环靠 answer 非空中断流程）：
        分支A 已确认   → (confirmed, "")        继续走检索
        分支B 有候选   → ([], 追问话术)          main_graph 路由到 node_answer_output 反问用户
        分支C 无结果   → ([], 拒识话术)          同上
    """
    confirmed = align_result.get("confirmed_item_names", []) or []
    options = align_result.get("options", []) or []

    if confirmed:
        return list(confirmed), ""

    if options:
        options_str = "、".join(options[:OPTIONS_DISPLAY_N])
        return [], f"您是想问以下哪个产品：{options_str}？请明确一下型号。"

    return [], "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：校验参数
        # 经过校验以后  原始问题和session_id 会话ID
        session_id, original_query = self._step_1_validate_param(state)
        logger.info("步骤1：参数校验通过")

        # 步骤2：获取历史记录  MongoDB 读历史失败只降级为空历史（副作用存储，不打挂查询图）
        history = self._safe_get_recent_messages(session_id)
        logger.info(f"步骤2：获取到 {len(history)} 条历史消息")
        # 更新状态
        state["history"] = history  # 很重要

        # 步骤3：用户初始消息保存   "BrotherHAK180烫金机如何调节温度？"
        # 保存用户这一次 问题（写失败降级：message_id=None，步骤8 会改为补插入）
        message_id = self._safe_save_user_message(session_id, original_query)

        # 步骤4：提取信息  LLM 识别(商品名称)以后 消息历史
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res.get("item_names") or []  # 第四步 识别商品名称
        # rewritten_query 已在纯函数层保证非空（兜底原始问题），不会把空串传给下游检索
        rewritten_query = extract_res.get("rewritten_query") or original_query
        # 更新状态 状态
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names

        # 5. & 6. 如果有提取到商品名，进行搜索和对齐
        align_result: Dict[str, Any] = {}
        if len(item_names) > 0:
            query_results = self._step_5_vectorize_and_query(item_names)
            align_result = self._step_6_align_item_names(query_results)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        # 7. 检查确认状态
        state = self._step_7_check_confirmation(state, align_result, history)

        # 8. 写入最终历史（内部全部 try/except，Mongo 故障不影响本次查询返回）
        self._step_8_write_history(state, session_id, rewritten_query, message_id)
        try:
            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        except Exception as e:  # noqa: BLE001  # 打点失败是纯展示副作用，不能让节点被判失败
            logger.warning(f"节点完成打点失败（不影响检索）：{e}")
        return state

    # ---------------- MongoDB 安全包装（历史是副作用，故障必须降级） ----------------

    def _safe_get_recent_messages(self, session_id: str) -> List[Dict]:
        """
        读取历史会话：工具函数内部虽有兜底，但 Mongo 客户端构造失败仍可能上抛，
        这里再包一层——查不到历史就当新会话处理（LLM 少了上下文，但检索链路不断）。
        """
        try:
            return get_recent_messages(session_id) or []
        except Exception as e:  # noqa: BLE001
            logger.warning(f"读取Mongo历史失败，按无历史继续：{e}")
            return []

    def _safe_save_user_message(self, session_id: str, original_query: str) -> str | None:
        """
        保存用户初始消息：Mongo 不可用时返回 None，后续步骤8 会以"补插入"方式再尝试，
        绝不让整条查询图因为写历史失败而失败。
        """
        try:
            return save_chat_message(session_id, "user", original_query)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MongoDB 保存用户初始消息失败（流程继续）：{e}")
            return None

    def _step_7_check_confirmation(self, state, align_result, history):
        """
        7 检查step6对齐后的商品名状态，分3种分支更新state，并同步更新历史消息的商品名关联
        :param state: 字典 - 原始会话状态，包含session_id/original_query等核心字段
        :param align_result: 字典 - step6的对齐结果
        :param history: 列表[字典] - 近期会话历史
        :return: 字典 - 更新后的会话状态，包含item_names/answer
        """
        # 分支判定交给纯函数：A=已确认继续检索；B=候选追问；C=无结果拒识
        # B/C 的追问/拒识话术写入 state["answer"]，main_graph 的
        # _route_after_item_name_confirm 检测到 answer 非空即路由到 node_answer_output，
        # 把话术发给用户并结束本轮（前端 chat.html 消费 answer 完成交互闭环）
        confirmed_names, answer = resolve_confirmation(align_result)
        confirmed = align_result.get("confirmed_item_names", []) or []

        # 分支A：有确认的商品名（高置信度，无需用户确认）
        if confirmed_names:
            # 收集历史消息中未关联商品名的消息ID（需批量更新关联）
            ids_to_update = []
            for msg in history:  # 找到没有商品名称的  对应ID
                if not msg.get("item_names"):  # 仅更新item_names为空的历史消息
                    mid = msg.get("_id")  # 提取消息唯一ID
                    if mid:
                        ids_to_update.append(str(mid))  # 转为字符串，避免ID格式问题

            # 若存在需更新的消息ID，批量更新历史消息的商品名关联（主体、关键词）
            # Mongo 写失败仅 WARNING：回填关联是锦上添花，不能打挂已确认的主流程
            if ids_to_update:
                try:
                    update_message_item_names(ids_to_update, confirmed)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"MongoDB 回填历史消息商品名失败（流程继续）：{e}")

            # 更新会话状态：设置确认商品名、清空 answer（放行后续检索）
            state["item_names"] = confirmed_names
            state["answer"] = ""
            return state

        # 分支B/C：无确认商品名 → answer 已是追问话术或拒识话术，item_names 清空
        state["answer"] = answer
        state["item_names"] = []
        return state

    def _step_6_align_item_names(self, query_results) -> dict:
        """
        6 根据Milvus搜索评分，逐个对齐step4提取的item_names，生成「确认商品名」和「候选商品名」
        规则与实现见纯函数 align_item_names（含别名行去重、保序去重）。
        :param query_results: 列表[字典] - step5的返回结果
        :return: {"confirmed_item_names": [...], "options": [...]}
        """
        logger.info(f"步骤6：获得待处理的数据源：{query_results}")
        return align_item_names(query_results)

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        """
         8 把本次处理的核心信息（用户问题、助手答案、商品名、改写查询）写入MongoDB的会话历史
         包含2个核心操作：1. 写入助手答案（若有）；2. 更新用户原始问题的关联信息
         两处写入都包 try/except：历史写入只是副作用，Mongo 故障时降级为 WARNING，
         保证用户仍能拿到本次查询结果。
         :param state: 字典 - step7更新后的会话状态，包含answer/item_names等字段
         :param session_id: 字符串 - 会话唯一标识
         :param rewritten_query: 字符串 - step4改写后的完整问题
         :param message_id: 字符串 - 本次用户问题的消息唯一ID（步骤3失败时为 None）
         :return:
         """
        # 若会话状态中有助手答案（分支B/C），写入助手消息到历史
        if state.get("answer"):
            try:
                save_chat_message(
                    session_id=session_id,  # 会话ID，关联所属会话
                    role="assistant",  # 消息角色：助手
                    text=state["answer"],  # 消息内容：向用户确认的提示语/无结果提示语
                    rewritten_query="",  # 助手消息无需改写查询，设为空
                    item_names=state.get("item_names", [])  # 关联的商品名列表（分支B/C均为空）
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"MongoDB 写入助手追问消息失败（不影响本次返回）：{e}")

        # 强制更新本次用户原始问题的关联信息（核心：补充改写查询、商品名）
        # message_id 为 None（步骤3写入失败）时改为补插入一条完整用户消息，尽量不丢历史
        try:
            save_chat_message(
                session_id=session_id,  # 会话ID，关联所属会话
                role="user",  # 消息角色：用户
                text=state["original_query"],  # 消息内容：用户原始查询
                rewritten_query=rewritten_query,  # 补充step4改写后的完整问题
                item_names=state.get("item_names", []),  # 补充关联的商品名列表
                message_id=message_id  # 有ID→更新；None→新增（补插入）
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MongoDB 更新用户消息关联信息失败（不影响本次返回）：{e}")

        # 返回最终会话状态，供下游节点使用
        return state

    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        """
           把分析出的item_names逐个向量化（BGEM3模型），并在Milvus向量数据库(kb_item_names)中执行混合搜索，获取匹配评分
           :param item_names: 列表[字符串] - 步骤4中 提取的商品名列表（如["苹果15", "华为P60"]）
           :return: 列表[字典] - [{"extracted_name": ..., "matches": [{"item_name":..., "score":...}]}]
        """
        # 1、初始化最终返回结果列表，存储每个商品名的向量化查询结果
        results = []

        # 2、获取Milvus向量数据库的客户端连接对象
        # get_milvus_client 连不上会抛异常：这里捕获后返回空结果，
        # 让对齐走分支C（提示用户换个说法），而不是把整条查询图打挂
        try:
            client = get_milvus_client()
        except Exception as e:  # noqa: BLE001
            logger.error(f"连接 Milvus 失败，跳过商品名向量对齐：{e}")
            return results
        if not client:
            logger.error("连接 Milvus 失败")
            return results

        # 3、Milvus中存储商品名称向量的集合名（表名）；主键字段是 pk（milvus_utils 已动态适配）
        collection_name = milvus_config.item_name_collection  # kb_item_names

        # 4、对所有商品名称批量生成BGEM3向量（稠密+稀疏），相比逐个生成提升处理效率
        # 模型加载/推理失败同样降级为空结果（下游走"未找到相关产品"分支）
        try:
            embeddings = generate_embeddings(item_names)
        except Exception as e:  # noqa: BLE001
            logger.error(f"商品名向量化失败（BGE-M3），跳过向量对齐：{e}")
            return results

        # 5、遍历每个商品名称，逐个执行向量搜索（保证结果与原始商品名一一对应）
        for i in range(len(item_names)):
            try:
                # 当前商品名的稠密向量（1024维）与稀疏向量（{token_id: weight}）
                dense_vector = embeddings["dense"][i]
                sparse_vector = embeddings["sparse"][i]

                # 构造Milvus混合搜索请求对象，返回Top5匹配结果
                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=CANDIDATE_TOP_N
                )

                # 执行BGEM3混合向量搜索：稠/稀疏权重0.8/0.2，开启评分归一化（0-1）
                # hybrid_search 已把 3.x 返回结构归一化为 list[list[dict]]（失败返回 None）
                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2),
                    limit=CANDIDATE_TOP_N,
                    norm_score=True,  # 3.x WeightedRanker 默认 True，这里显式传保证两版本行为一致
                    output_fields=["item_name"]
                )

                # 收集当前商品名的匹配结果（别名行在 step6 去重，这里只做原样收集）
                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        matches.append(
                            {
                                "item_name": (hit.get("entity") or {}).get("item_name"),
                                "score": hit.get("distance"),
                            }
                        )

                results.append({
                    "extracted_name": item_names[i],
                    "matches": matches
                })

            # 捕获单个商品名处理的异常（不中断其他商品名执行），仅记录错误日志
            except Exception as e:  # noqa: BLE001
                logger.error(f"查询商品名 '{item_names[i]}' 时出错: {e}")

        # 返回所有商品名的向量化+搜索结果列表
        return results

    def _step_4_extract_info(self, query, history) -> Dict:
        """
        利用LLM从当前问题以及历史会话中提取出主要询问的商品名称item_names（可多个，JSON列表形式）
        若商品名不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
        :param query: 字符串 - 用户当前原始查询问题（如："这个多少钱？"）
        :param history: 列表[字典] - 近期会话历史
        :return: {"item_names": [...], "rewritten_query": "..."}（rewritten_query 保证非空）
        """

        try:
            # 1. 获取llm客户端
            # 模型档位走配置 item_model（本机=deepseek-flash）；
            # 禁止硬编码 deepseek-chat/deepseek-reasoner——该账号只有 flash/v4-pro 两档
            chat_model = ChatOpenAI(
                model=lm_config.item_model,
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.llm_temperature,
                # 开启JSON标准输出模式，强制模型返回可解析的json_object
                model_kwargs={
                    "response_format": {"type": "json_object"}
                }
            )

            # 2. 构造历史对话文本，拼接为"角色: 内容"的格式，供LLM做上下文理解
            history_text = self._build_history_text(history)

            # 3. 处理和动态拼接提示词
            # 模板用命名占位符 {history_text}/{query}，与 format 的关键字参数一一对应；
            # 模板里的 JSON 花括号已用 {{}} 转义为普通字符
            user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_text,
                query=query
            )

            # 4. 构造LLM调用的消息列表，包含系统角色（定义助手身份）和用户角色（传入提示词）
            messages = [
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]

            # 5. 调用LLM客户端，发起请求获取结果
            response = chat_model.invoke(messages)

            # 6. 取正文：content 为空试 reasoning_content，全空在这里抛错走兜底
            raw_text = extract_llm_text(response)

            # 7. 修复解析：围栏/前置文本/尾逗号/单引号/截断 全部交给 json_format_utils，
            #    脏 JSON 不再用裸 json.loads 直接炸掉本次提取
            result = parse_item_names_result(raw_text, query)
            logger.info(f"步骤4：提取结果 item_names={result['item_names']}, "
                        f"rewritten_query={result['rewritten_query']}")
            return result

        except Exception as e:  # noqa: BLE001
            # 捕获所有异常（LLM 调用失败、返回全空、JSON 彻底修不动等），记录错误日志
            logger.error(f"大模型调用/解析异常：{e}")
            # 异常兜底：原始问题原样传下去 + 空商品名。
            # 绝不能返回空 rewritten_query——它会被写进 state 并被下游检索节点使用，
            # 空改写等于把所有检索意图抹掉，整条链路结果全废
            return {"item_names": [], "rewritten_query": query}

    @staticmethod
    def _build_history_text(history) -> str:
        """把历史消息列表拼接为 "role: text" 多行文本（无历史返回空串）。"""
        lines = []
        for msg in history or []:
            role = msg.get("role") or ""
            content = msg.get("text") or ""
            lines.append(f"{role}: {content}")
        return "\n".join(lines) + ("\n" if lines else "")

    def _step_1_validate_param(self, state: QueryGraphState) -> Tuple[str, str]:

        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心参数original_query缺失")

        return session_id, original_query


if __name__ == "__main__":
    # 本地联调入口（需要 Mongo/Milvus/DeepSeek 全部就绪才能跑通，平时由主控串行验证）
    init_state = {
        "session_id": "test_session_010",  # 会话ID
        "original_query": "HAK 180 烫金机怎么用？"
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()  # __call__
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 输出（format_json 支持 ObjectId 序列化）
    from utils.json_format_utils import format_json

    logger.info(format_json(result))
