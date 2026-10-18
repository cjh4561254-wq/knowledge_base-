# processor/query_processor/nodes/node_answer_output.py

import re
from typing import Any, Dict, List, Optional, Tuple

from processor.query_processor.prompt.answer_prompt import ANSWER_PROMPT
from tool.logger import logger

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from utils.llm_utils import get_llm_client
from utils.mongo_history_utils import save_chat_message
from utils.sse_utils import push_to_session, SSEEvent
from utils.task_utils import set_task_result, add_done_task

# 拼进 prompt 的最大字符预算：防止长文档把 LLM 上下文"撑死"（token 溢出）
MAX_CONTEXT_CHARS = 12000

# 空召回兜底答案：reranked_docs 为空时直接输出该文案，绝不把空 prompt 送进 LLM（避免幻觉）
NO_RECALL_ANSWER = "抱歉，知识库中未检索到与该问题相关的内容，无法为您回答。您可以尝试更换商品型号或补充相关文档后重试。"

# LLM 调用失败时的降级答案
LLM_ERROR_ANSWER = "抱歉，生成回答时出现错误，请稍后重试。"


def extract_chunk_text(chunk: Any) -> str:
    """
    从 LLM 流式 chunk / 完整 response 中容错地提取可读文本（纯函数，便于单测）。

    为什么要容错：
    - DeepSeek 思考模型（如 deepseek-v4-pro）的 chunk 里 content 与 reasoning_content
      可能单独为空：多数 chunk 只有 reasoning_content，最后才有 content；
      两侧都可能为空，直接取 .content 会得到 None 并拼出 'None' 或抛错。
    - langchain 1.x 部分 provider 的 AIMessageChunk.content 是内容块列表
      （[{"type":"text","text":...}]），不是纯字符串。
    策略：优先正文 content（str 或块列表拼接），为空时回退 reasoning_content 兜底，
    保证流不中断、有 token 就能展示。
    """
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk

    content = getattr(chunk, "content", None)
    if content is None and isinstance(chunk, dict):
        content = chunk.get("content")

    if isinstance(content, str) and content:
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        text = "".join(parts)
        if text:
            return text

    # content 为空：回退 reasoning_content（属性 / additional_kwargs / dict 三种携带方式）
    reasoning = getattr(chunk, "reasoning_content", None)
    if reasoning is None:
        additional = getattr(chunk, "additional_kwargs", None)
        if isinstance(additional, dict):
            reasoning = additional.get("reasoning_content")
    if reasoning is None and isinstance(chunk, dict):
        reasoning = chunk.get("reasoning_content")

    return str(reasoning) if reasoning else ""


def _first_text(*values: Any) -> str:
    """
    取第一个非空文本（None / 纯空白都算没取到），统一返回 str（取不到返回空串）。
    为什么：引用来源的 title/url 一旦为 None，前端会把 "None" 渲染出来；
    title 还需按 title → file_title → parent_title → item_name 兜底（与 node_rerank 的取值链一致），
    因为 kb_chunks 的标题可能整字段缺失或为空串（旧数据 / 检索未带 output_fields 的历史召回）。
    """
    for value in values:
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        if text.strip():
            return text
    return ""


def build_sources(docs: List[Dict]) -> List[Dict[str, Any]]:
    """
    把真正进入 prompt 的重排文档转成前端可渲染的来源列表（纯函数）。

    编号契约（与 _format_reranked_docs 共用同一套编号源，方案 A）：
    - 入参 docs 必须是 _format_reranked_docs 返回的 included_docs（已按相关性有序，
      且只含真正写进 prompt 的切片：空正文切片与预算截断掉的尾部切片都不在列表里）；
    - 因此这里 enumerate(start=1) 得到的 idx 与 prompt 元数据头里的 [N] 严格相等，
      答案文本中的 [doc:N] 与 sources[].index / sources[].ref（"[N]"）一一对应；
    - 前端 chat.html 按 index 精确等值匹配，不按位置取 sources[N-1]，所以 index 与 ref
      必须同步由同一个 idx 派生，任何一侧改号都要改另一侧。
    字段 None 统一归一化为空串，避免前端渲染出 "None" 字样。
    字段契约：{"index","ref","chunk_id","source","title","url","score"}（不可改名）。
    """
    sources: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        entity = doc.get("entity") if isinstance(doc.get("entity"), dict) else {}
        sources.append(
            {
                "index": idx,
                "ref": f"[{idx}]",
                "chunk_id": "" if doc.get("chunk_id") is None else str(doc.get("chunk_id")),
                "source": "" if doc.get("source") is None else str(doc.get("source")),
                # 标题兜底链：切片标题 → 源文件名 → 父标题 → 商品主体名 → entity 里的同名字段
                # （node_rerank 已做过一次兜底，这里是最后一道防线：
                #  上游 rerank 异常降级时 reranked_docs 可能带的是原始字段结构）
                "title": _first_text(
                    doc.get("title"),
                    doc.get("file_title"),
                    doc.get("parent_title"),
                    doc.get("item_name"),
                    entity.get("title"),
                    entity.get("file_title"),
                    entity.get("parent_title"),
                    entity.get("item_name"),
                ),
                "url": "" if doc.get("url") is None else str(doc.get("url")),
                "score": doc.get("score"),
            }
        )
    return sources


def assemble_final_payload(
    answer: str,
    image_urls: Optional[List[str]],
    item_names: Optional[List[str]],
    sources: Optional[List[Dict]],
) -> Dict[str, Any]:
    """
    组装 final 事件 / 非流式结果的统一 payload（纯函数）。
    字段契约（与 utils/sse_utils.py 及前端 chat.html 对齐）：
    - answer     : 完整答案文本（前端优先用它渲染，其次回退累计的 delta）
    - status     : 固定 "completed"
    - image_urls : 从召回文档提取的配图 URL 列表（前端 renderAnswerWithImages 消费）
    - item_names : 本轮确认的商品名列表
    - sources    : 来源文档列表（引用标注，index 与 prompt 中的 [N] 编号一致）
    """
    return {
        "answer": answer or "",
        "status": "completed",
        "image_urls": image_urls or [],
        "item_names": item_names or [],
        "sources": sources or [],
    }


class NodeAnswerOutput(NodeBase):
    """
    节点功能: 答案输出
    流程: 检查已有答案 →（空召回兜底 | 构建提示词 → LLM 生成）→ 写入历史 → 发送结束事件
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
        2 根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织 prompt 并调用llm 生成答案
        3 调用大模型输出答案 注意判断是否需要流式输出需要则流式输出
        4 把答案写入到mongodb的history中 利用utils/mongo_history_utils.py中的save_chat_message方法
        5 做最后一次push操作（主要是为了触发前端图片渲染)
             {
                "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
                "status": "completed",
                "image_urls": [...],
                "item_names": [...],
                "sources": [...]
              }
        """
        # 阶段一：检查answer是否存在,如果存在直接输出answer中的答案
        answer_exists = self._step_1_check_answer(state)

        # 本轮真正生成的答案（含空召回兜底）才需要写历史；
        # 上游确认节点（反问/拒答）的答案已在 node_item_name_confirm 落库，这里再写会重复
        generated_here = False
        sources: List[Dict] = []

        # 阶段二：如果没有answer则 构建 Prompt（空召回走兜底答案，不进 LLM，防幻觉）
        if not answer_exists:
            reranked_docs = state.get("reranked_docs") or []
            if not reranked_docs:
                logger.warning("---Step 2: reranked_docs 为空，使用空召回兜底答案，跳过 LLM 调用---")
                self._emit_fallback_answer(state, NO_RECALL_ANSWER)
            else:
                prompt, included_docs = self._step_2_construct_prompt(state, reranked_docs)
                state["prompt"] = prompt
                # 引用标注：只有真正进入 prompt 的文档才算可引用来源
                sources = build_sources(included_docs)
                # 阶段三：调用大模型输出答案（流式/非流式）
                self._step_3_generate_response(state, prompt)
            generated_here = True

        # 阶段四：提取图片URL（用于历史记录和前端展示）
        image_urls = self._extract_images_from_docs(state.get("reranked_docs") or [])

        # 阶段五：把答案写入到mongodb的history中（用户问题已由确认节点落库，这里只写助手答案）
        if generated_here and state.get("answer"):
            logger.info("---写入MongoDB历史记录---")
            # sources 一起落库：刷新页面走 /history 还原时才能把 [doc:N] 角标链回原文，
            # 否则历史消息只能降级成纯文本（前端 normalizeSources 拿到空表会删标记）
            self._step_4_write_history(state, image_urls=image_urls, sources=sources)

        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))

        # 阶段六: 流式输出结束，发送 final 事件 [最后兜底，确保图片都能渲染和结束]，
        # 再补发 CLOSE 内部信号让 sse_generator 下发 done 终止帧并清理队列
        logger.info(f"---发送 final 事件---图片为：{image_urls}")
        if state.get("is_stream"):
            session_id = state.get("session_id")
            payload = assemble_final_payload(
                state.get("answer") or "", image_urls, state.get("item_names") or [], sources
            )
            push_to_session(session_id, SSEEvent.FINAL, payload)
            # CLOSE 是内部事件：sse_generator 收到后补发 done 终止帧并退出（前端 EventSource
            # 在 final 已 close()，done 帧是无监听者兜底，不会造成重复渲染）
            push_to_session(session_id, SSEEvent.CLOSE, {})

        logger.info("---node_answer_output 节点处理结束---")

        return state

    def _step_1_check_answer(self, state: QueryGraphState) -> bool:
        """
        阶段一：检查 state 中是否已有 answer（来自商品确认节点的反问/拒答）。
        - 若已存在：流式按需推送 delta，并写 task result 供 web 层读取，返回 True
        - 若不存在：返回 False
        """
        answer = state.get("answer")
        is_stream = state.get("is_stream")
        if answer:
            session_id = state.get("session_id")
            if is_stream:
                logger.info("---Step 1: 发现已有答案，执行流式推送---")
                push_to_session(session_id, SSEEvent.DELTA, {"delta": answer})
            # 无论流式/非流式都写 result：非流式靠它取 answer，流式作为兜底快照
            set_task_result(session_id, "answer", answer)
            return True
        return False

    def _emit_fallback_answer(self, state: QueryGraphState, text: str) -> None:
        """
        空召回兜底：不调用 LLM，直接产固定答案，并按 is_stream 走与正常输出相同的事件序列
        （流式推一条 DELTA + 写 result；FINAL 由阶段六统一发送）。
        """
        state["answer"] = text
        session_id = state.get("session_id")
        if state.get("is_stream"):
            push_to_session(session_id, SSEEvent.DELTA, {"delta": text})
        set_task_result(session_id, "answer", text)

    def _step_2_construct_prompt(
        self, state: QueryGraphState, reranked_docs: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        阶段二：构建 Prompt
        根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组装 LLM 提示词。
        返回 (prompt, 实际进入上下文的文档列表) —— 后者用于生成引用来源(sources)列表。
        """
        char_budget = MAX_CONTEXT_CHARS

        # 1. 获取问题和商品名
        # 优先使用重写后的问题
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names") or []

        # 2. 格式化上下文文档（受字符预算约束，超出截断并记日志）
        context_str, char_budget, included_docs = self._format_reranked_docs(
            reranked_docs, char_budget
        )

        # 3. 格式化历史对话
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        # 4. 格式化 Item Names (提问商品) 列表转字符串
        item_names_str = ", ".join(item_names) if item_names else "无指定商品"

        # 5. 组装提示词（ANSWER_PROMPT 的 4 个占位符 context/history/item_names/question 与此一一对应）
        prompt = ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=item_names_str,
            question=question,
        )
        logger.info(f"组装后的提示词为：{prompt}")
        return prompt, included_docs

    def _format_reranked_docs(
        self, reranked_docs: List[Dict], char_budget: int
    ) -> Tuple[str, int, List[Dict]]:
        """格式化重排序文档，带字符预算控制。

        返回 (拼接后的上下文字符串, 剩余预算, 实际纳入的文档列表)。
        """
        formatted_lines: List[str] = []
        included_docs: List[Dict] = []
        used_chars = 0

        # 从重排内容中，提取为资料字符串，不可超过限额
        # 优先使用结构化 reranked_docs（包含 source/chunk_id/url/score），便于约束与引用
        # ---------------------------------------------------------
        # 逻辑解释：
        # 1. 遍历重排序后的文档列表 (reranked_docs)，这些文档已经按相关性从高到低排序。
        # 2. 对每个文档提取关键信息 (content, source, chunk_id, url, title, score)。
        # 3. 构造 "元数据头 + 正文" 格式的字符串，例如：
        #    "[1] [source=local] [chunk_id=123] [score=0.9500] [title=操作手册]
        #     这里是文档的正文内容..."
        # 4. 累加字符长度，如果超过 MAX_CONTEXT_CHARS (如 12000 字符)，则停止添加并记日志，
        #    确保 Prompt 长度在 LLM 的处理范围内，避免 Token 溢出。
        # ---------------------------------------------------------
        # 编号口径（修复点，方案 A）：编号 = 该文档在 included_docs 中的位次（len+1），
        # 与 build_sources 对同一列表 enumerate(start=1) 完全同源，保证「prompt 里的 [N]」
        # 与「sources[].index」严格相等。
        #   为什么选 A 而不是 B（B = build_sources 沿用 reranked_docs 原始位次、允许稀疏 index）：
        #   1) 改动面最小：只动本循环的取号表达式，build_sources 的参数顺序/返回结构
        #      （web/api/query_service.py 会 import 复用）与字段契约都保持原样；
        #      B 方案要把原始编号回传给 build_sources，只能往 doc 里塞字段（污染 state 的
        #      reranked_docs）或给 build_sources 加参数，都属于改签名。
        #   2) 不产生语义漂移：库里/接口上已有的 sources 一直是「从 1 连续」的形态，
        #      前端 normalizeSources 也按 index 精确匹配；A 保持连续，历史数据与新数据同构，
        #      B 会让新数据出现 index 跳号（1、3），与旧记录形态不一致。
        #   3) 空正文切片根本没进 prompt，模型看不到它，给它留号反而会让答案里出现
        #      「模型未读过的编号」，跳号也毫无收益。
        #   注：idx（reranked_docs 原始位次）仍用于预算截断日志，与引用编号无关。
        for idx, doc in enumerate(reranked_docs, start=1):
            # 内容为空的切片不进 prompt：既省预算也防止 LLM 对着空引用编造事实
            # 这里 continue 不消耗编号 —— 编号只在真正写入 prompt 时才递增
            content = (doc.get("content") or "").strip()
            if not content:
                continue

            ref_no = len(included_docs) + 1
            meta_tags = [f"[{ref_no}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                raw_value = doc.get(field)
                # None/空值直接跳过：此前 str(None) 会把 "None" 拼进元数据头
                field_value = "" if raw_value is None else str(raw_value).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                try:
                    meta_tags.append(f"[score={float(relevance_score):.4f}]")
                except (TypeError, ValueError):
                    # score 是异常类型时舍弃该标签，不能让格式化炸掉整条链路
                    pass

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                logger.warning(
                    f"上下文预算耗尽截断：已纳入 {len(included_docs)} 篇（{used_chars}/{char_budget} 字符），"
                    f"丢弃剩余 {len(reranked_docs) - idx + 1} 篇召回文档"
                )
                break

            formatted_lines.append(doc_entry)
            included_docs.append(doc)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars, included_docs

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"
            used_chars += len(formatted_line) + 1

            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars

            formatted_lines.append(formatted_line)

        return "\n".join(formatted_lines), char_budget - used_chars

    def _step_3_generate_response(self, state: QueryGraphState, prompt: str) -> QueryGraphState:
        """
        阶段三：生成回答
        调用llm生成答案，支持流式输出；流式取值走 extract_chunk_text 容错
        （content 与 reasoning_content 都可能单独为空）。
        """
        logger.info("---Step 3: 开始生成回答 (LLM Generation)---")

        # 判断是否需要流式输出
        # 通常 state 中会注入 stream_queue 用于 SSE 推送
        session_id = state.get("session_id")
        is_stream = state.get("is_stream")

        # 统一取 LLM 客户端（内部带缓存，非重操作）
        llm = get_llm_client()

        if is_stream:
            logger.info(f"模式: 流式输出 (Streaming), Session: {session_id}")
            final_text = ""
            try:
                # 使用 stream 方法进行流式生成，逐 token 推 DELTA
                for chunk in llm.stream(prompt):
                    delta = extract_chunk_text(chunk)
                    if delta:
                        final_text += delta
                        # 将增量内容放入队列
                        push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})

                logger.info(f"流式输出完成，总长度: {len(final_text)}")

            except Exception as e:
                logger.error(f"流式生成出错: {e}", exc_info=True)
                # 发生错误时，尝试推送到前端；已产出的部分文本不丢弃
                push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
                if not final_text:
                    final_text = LLM_ERROR_ANSWER

            state["answer"] = final_text
            # 流式同样写 result：web 层 / 兜底逻辑可事后取完整答案
            set_task_result(session_id, "answer", final_text)
        else:
            # 非流式直接调用
            logger.info(f"模式: 非流式输出 (Blocking), Session: {session_id}")
            try:
                response = llm.invoke(prompt)
                content = extract_chunk_text(response)
                state["answer"] = content
                set_task_result(session_id, "answer", content)
                logger.info(f"生成回答完成，长度: {len(content)}")
            except Exception as e:
                logger.error(f"生成回答出错: {e}", exc_info=True)
                state["answer"] = LLM_ERROR_ANSWER
                set_task_result(session_id, "answer", LLM_ERROR_ANSWER)

        return state

    def _extract_images_from_docs(self, docs: List[Dict]) -> List[str]:
        """
        辅助方法：从文档列表中提取图片URL

        核心逻辑：
        1. 遍历所有相关文档（包括本地知识库切片和联网搜索结果）。
        2. 策略一：直接检查文档的 'url' 字段（常见于联网搜索结果）。
           - 验证后缀名是否为图片格式 (.jpg, .png 等)。
        3. 策略二：使用正则表达式扫描文档 'content' 正文内容（常见于本地 Markdown 文档）。
           - 匹配 Markdown 图片语法: ![alt text](image_url)。
        4. 对提取到的 URL 进行去重处理，返回唯一图片列表。

        :param docs: 文档列表，每个文档为字典格式
        :return: 图片 URL 字符串列表
        """
        images = []
        seen = set()  # 用于去重，避免同一张图片重复出现
        if not docs:
            return []
        # ---------------------------------------------------------
        # 正则表达式解释：r'!\[.*?\]\((.*?)\)'
        # 1. !\[   -> 匹配 Markdown 图片语法的开头 "![" (注意 [ 需要转义)
        # 2. .*?   -> 非贪婪匹配图片描述文本 (Alt Text)，即 [] 中间的内容
        # 3. \]    -> 匹配描述文本的结束符 "]"
        # 4. \(    -> 匹配 URL 部分的开始符 "("
        # 5. (.*?) -> 捕获组 (Group 1)：非贪婪匹配括号内的实际 URL 内容
        # 6. \)    -> 匹配 URL 部分的结束符 ")"
        # ( ... ) （不带反斜杠）：这就是 捕获组 。
        # 它的作用是告诉程序：“虽然我匹配了整个 ![...](...) 结构，但我 只要 这括号里的内容”。
        # ---------------------------------------------------------
        md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

        logger.info(f"开始提取图片，待处理文档数: {len(docs)}")

        for i, doc in enumerate(docs):
            # 1. 优先检查 url 字段 (主要针对 Web Search 结果)
            url = (doc.get("url") or "").strip()
            if url:
                # 简单后缀判断：确保是静态图片资源
                if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                    if url not in seen:
                        logger.debug(f"文档[{i}] 发现图片 URL (字段): {url}")
                        seen.add(url)
                        images.append(url)

            # 2. 检查 content 字段中的 Markdown 图片 (主要针对 Local Chunk)
            text = (doc.get("content") or "").strip()
            if text:
                # findall 机制解释：
                # 正则表达式 r'!\[.*?\]\((.*?)\)' 中包含一个捕获组 (.*?)
                # 当存在捕获组时，findall 只返回括号内匹配到的内容（即 URL），而不是整个 ![...](...) 字符串
                # 示例：
                # 输入 text: "参考图片 ![面板图](http://img.com/1.jpg) 如下"
                # 返回 matches: ['http://img.com/1.jpg']
                matches = md_img_pattern.findall(text)
                for img_url in matches:
                    img_url = img_url.strip()
                    if img_url and img_url not in seen:
                        logger.debug(f"文档[{i}] 正文发现 Markdown 图片: {img_url}")
                        seen.add(img_url)
                        images.append(img_url)

        logger.info(f"图片提取完成，共找到 {len(images)} 张唯一图片: {images}")
        return images

    def _step_4_write_history(
        self, state: QueryGraphState, image_urls=None, sources=None
    ) -> QueryGraphState:
        """
        阶段四：把本轮答案写入 MongoDB history（用户问题已由 node_item_name_confirm 落库）。
        Mongo 抖动只降级为 WARNING，不能让已生成的答案因写库失败而丢失/抛出。
        :param sources: build_sources(included_docs) 的产物，与答案里的 [doc:N] 编号同源；
                        只写 assistant 消息（user 消息没有引用来源），老记录缺该键由读取侧 or [] 兜底。
        """
        session_id = state.get("session_id", "default")
        answer = (state.get("answer") or "").strip()
        item_names = state.get("item_names") or []
        rewritten_query = state.get("rewritten_query") or ""

        try:
            if answer:
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    rewritten_query=rewritten_query,
                    item_names=item_names,
                    image_urls=image_urls,
                    message_id=None,
                    # 新参数在末尾且带默认值：其它调用方（node_item_name_confirm 等）无需改动
                    sources=sources,
                )
        except Exception as e:
            # 写历史失败不应影响主链路（答案已生成并推送给用户）
            logger.warning(f"写入Mongo历史记录失败（已降级，不影响答案输出）: {e}")

        return state


if __name__ == "__main__":

    logger.info("开始测试: 答案生成节点")

    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2",
            "score": 0.95,
            "url": None,
            "content": """
HAK 180 烫金机的操作面板位于机器正前方。
开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
具体的操作面板布局请参考下图：
![操作面板布局图](http://192.168.100.100:9000/knowledge-base/upload-images/hak180产品安全手册/048c005b198be5c9fff80ad6a6ba02496f38fa109ec20dbaabde3110f3eb1574.jpg)

如果是进行局部烫金，请调节侧面的旋钮。
![侧面旋钮细节](http://192.168.100.100:9000/knowledge-base/upload-images/hak180产品安全手册/f77da4df52517fc50b9efb528540e1351dd1a08dce6f801cf08366540f2c59ce.jpg)
"""
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://192.168.100.100:9000/knowledge-base/upload-images/example.jpg",  # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "content": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "url": None,
            "content": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK180烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None  # 初始无答案
    }

    # 运行节点
    node_answer_output = NodeAnswerOutput()
    result = node_answer_output(mock_state)

    print(result)
