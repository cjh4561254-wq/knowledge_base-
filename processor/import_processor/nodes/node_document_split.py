# processor/import_processor/nodes/node_document_split.py
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Tuple, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState

# ==================== 模块级正则与常量（切片边界保护用） ====================
# 代码围栏起始：3个及以上反引号/波浪线，允许携带语言信息串如 ```python（课程v2+CommonMark口径）
_CODE_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
# Markdown表格行：以|开头并以|结尾（含对齐分隔行 |---|---|）
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
# 句子结束标点：中英文全部覆盖（句子级切分与重叠共用同一套判定，保证切/叠口径一致）
_SENTENCE_END_CHARS = "。？！；.!?;\n"


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点：文档切分（node_document_split）
        整体流程：加载输入→按MD标题初切→长切短合→统计输出→结果备份
        核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
        后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等

        必要参数：md_path(完整流程中非必要，备份测试用的json文件)、md_content、file_title
        更新参数：chunks

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # ===================================== 步骤1：加载并标准化输入数据 =====================================
        # 作用：从状态字典提取MD内容/文件标题，统一换行符消除系统差异
        # 输出：标准化后的md_content、文件标题；
        content, file_title = self._step_1_get_inputs(state)

        # ===================================== 步骤2：按MD标题进行初次切分 ===============================
        # 作用：基于Markdown标题（#/##/###）切分文档为独立章节，自动跳过代码块内的伪标题，保证章节语义完整
        # 输出：初切后的章节列表、识别到的有效标题数量、MD原始文本总行数（为后续统计/日志使用）
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)

        # ===================================== 步骤3：无标题场景兜底处理 ===================================
        # 作用：解决MD文档无任何标题的边界情况，避免后续切分逻辑异常
        # 输出：有标题则返回步骤2的章节列表；无标题则将全文封装为单个「无标题」章节，保证数据格式统一
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        # ===================================== 步骤4：Chunk精细化处理（长切短合） ===========================
        # 作用：核心切分逻辑，先将超长章节按「段落→句子」二次切分（表格/代码块作为原子块整体保留），
        #      再合并同父标题的过短章节，最后补齐parent_title兜底，减少碎片化
        # 输出：长度适中、语义完整、低碎片化的最终Chunk列表（可直接用于向量入库/大模型调用）
        sections = self._step_4_refine_chunks(sections)  # 片段过长 切  片段过短 合并

        # ===================================== 步骤4+：切片唯一身份标记 ==================================
        # 作用：给每个Chunk写入确定性的 chunk_index / chunk_uid，保证幂等重跑时可追踪、可对齐
        # 说明：Milvus主键 chunk_id 是 auto_id 生成后由入库节点回填的，此处刻意不叫chunk_id避免冲突
        sections = self._assign_chunk_identity(file_title, sections)

        # ===================================== 步骤5：输出文档切分统计信息 ==================================
        # 作用：打印核心统计数据，便于监控切分效果、调试问题（原始行数/最终Chunk数/首个Chunk预览）
        # 输出：无返回值，仅通过logger输出标准化统计日志
        self._step_5_print_stats(lines_count, sections)

        # ===================================== 步骤6：Chunk结果本地JSON备份 + 状态更新 ======================
        # 作用：将最终Chunk列表备份到local_dir目录的chunks.json，便于后续问题排查、数据复用
        # 输出：无返回值
        self._step_6_backup(state, sections)

        # 写入状态字典
        state["chunks"] = sections
        return state

    def _step_6_backup(self, state: ImportGraphState, sections: List[Dict[str, Any]]) -> None:
        """
        【步骤6】Chunk结果本地JSON备份（便于调试/问题排查，保留处理结果）

        原则（为什么这样分层捕获）：
            备份只是副作用，主产物永远是 state["chunks"]——任何备份失败都绝不能再吞掉
            或中断主切片结果；但「吞掉」不等于「静默」：失败必须以 WARNING/ERROR 级
            可见地写进日志，并说清后果（chunks.json 丢失，影响排障/复用，不影响本次导入）。
        两类失败分开处理：
            1) KeyError（state 里根本没有 md_path）：属上游状态传递问题，WARNING 级即可，
               单节点调试时无 md_path 是预期场景，不该升级成 ERROR 制造假警报；
            2) IO 错误（磁盘满/权限/路径非法等）：属环境问题，ERROR 级并带堆栈，
               否则运维永远看不到备份悄悄丢了。
        :param state: 项目状态字典，备份目录取自 md_path 的父目录
        :param sections: 最终处理后的Chunk列表
        """

        # ---------- 类别1：状态问题——md_path 字段缺失（单独一小段 try，KeyError 只可能在这里） ----------
        try:
            md_path = state["md_path"]
        except KeyError as e:
            # WARNING 而不是静默跳过：让「chunks.json 没落盘」这件事在日志里看得见
            self.logger.warning(
                f"步骤6：state 缺少 md_path 字段（KeyError: {e}，属状态传递问题），"
                f"本次 {len(sections)} 个Chunk的 chunks.json 备份已跳过——"
                f"后果：本地无切片备份可供排障/复用；主流程不受影响，chunks 照常写入 state"
            )
            return
        if not md_path:
            # 键存在但值为 None/空串：同样是状态问题，口径与 KeyError 分支保持一致
            self.logger.warning(
                f"步骤6：state['md_path'] 为空（状态传递问题），"
                f"本次 {len(sections)} 个Chunk的 chunks.json 备份已跳过——"
                f"后果：本地无切片备份可供排障/复用；主流程不受影响，chunks 照常写入 state"
            )
            return

        # ---------- 类别2：IO/环境问题——路径拼接与落盘（含非OSError的意外异常兜底） ----------
        try:
            # 拼接备份文件路径：固定文件名，便于查找（路径非法在这里也会落入下方 except）
            backup_path = Path(md_path).parent / "chunks.json"
            # 写入JSON文件：保留中文/格式化缩进，便于人工查看
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(
                    sections,
                    f,
                    # 开启 True："title": "\u4e00\u7ea7\u6807\u9898"（乱码，无法直接看）；
                    # 开启 False："title": "一级标题"（正常中文，人工可直接阅读）。
                    ensure_ascii=False,  # 保留中文，不转义为\u编码
                    indent=2  # 格式化缩进，便于阅读
                )
            self.logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")
        except OSError as e:
            # 磁盘/权限/目录不存在等环境问题：ERROR + 堆栈，但止步于此，不打断主流程
            self.logger.error(
                f"步骤6：chunks.json 写入失败（IO错误: {e}）——"
                f"后果：本地切片备份丢失，排障时将无 chunks.json 可查；"
                f"主流程不受影响，chunks 照常写入 state。备份目标：{md_path}",
                exc_info=True,
            )
        except Exception as e:
            # 非IO的意外异常（如 md_path 类型非法导致 Path 报错）：同样 ERROR 可见、主流程继续
            self.logger.error(
                f"步骤6：chunks.json 备份出现非IO意外异常（{type(e).__name__}: {e}）——"
                f"后果：本地切片备份丢失；主流程不受影响，chunks 照常写入 state。md_path={md_path!r}",
                exc_info=True,
            )

    def _step_5_print_stats(self, lines_count: int, sections: List[Dict[str, Any]]) -> None:
        """
        【步骤5】输出文档切分统计信息（日志记录，便于监控/调试）
        :param lines_count: MD原始文本总行数
        :param sections: 最终处理后的Chunk列表
        """
        chunk_num = len(sections)
        # 输出核心统计信息：原始行数/最终Chunk数
        self.logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        self.logger.info(f"MD原始文本总行数：{lines_count}")
        self.logger.info(f"最终生成Chunk数量：{chunk_num}")

    # ==================== 句子级工具（切分与重叠共用） ====================

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        【辅助函数】按中英文句末标点把文本拆成句子（保留标点）
        为什么：句子级重叠必须与句子级切分使用同一口径，否则重叠内容会错位
        :param text: 待拆分的纯文本
        :return: 句子列表（已去首尾空白，空句剔除）
        """
        sentences: List[str] = []
        buf: List[str] = []
        for ch in text:
            buf.append(ch)
            if ch in _SENTENCE_END_CHARS:
                sentence = "".join(buf).strip()
                if sentence:
                    sentences.append(sentence)
                buf = []
        tail = "".join(buf).strip()
        if tail:
            sentences.append(tail)
        return sentences

    @staticmethod
    def _is_atomic_like(sentence: str) -> bool:
        """
        【辅助函数】判断一个"句子"实际是表格行/代码围栏（这类内容不应作为重叠句复制）
        :param sentence: 单个句子
        :return: True表示不应参与重叠
        """
        if "```" in sentence or "~~~" in sentence:
            return True
        return bool(_TABLE_LINE_RE.match(sentence))

    def _pick_overlap_sentences(self, prev_sentences: List[str], budget: int) -> List[str]:
        """
        【辅助函数】从上一Chunk的句子尾部挑选重叠句
        规则：取最后 config.overlap_sentences 个「普通散文句」（跳过表格/代码行），
             并且总长不得超过剩余预算budget（保证Chunk不超max_content_length）
        :param prev_sentences: 上一个Chunk正文的句子列表
        :param budget: 重叠内容可用的字符数
        :return: 按原文顺序排列的重叠句列表
        """
        overlap_n = int(getattr(self.config, "overlap_sentences", 0) or 0)
        picked: List[str] = []
        if overlap_n <= 0 or budget <= 0:
            return picked
        used = 0
        for sentence in reversed(prev_sentences):
            if len(picked) >= overlap_n:
                break
            # 表格行/代码围栏整体重叠会污染语义且很长，直接跳过
            if self._is_atomic_like(sentence):
                continue
            cost = len(sentence) + 1  # +1 是句子间换行符
            if used + cost > budget:
                break
            picked.insert(0, sentence)
            used += cost
        return picked

    # ==================== 表格/代码块原子保护 ====================

    @staticmethod
    def _mask_atomic_blocks(body: str) -> Tuple[str, Dict[str, str]]:
        """
        【辅助函数】把正文中的代码块与表格替换成占位符（原子块保护）
        为什么：RecursiveCharacterTextSplitter会按\\n/句号继续细分，
              若不加保护，超长的代码块/表格会被从中间切断，导致语义破坏（检索出半截SQL/半张表）
        规则：1) 代码围栏从 ```/~~~ 起始行到配对结束行整体为一个原子块（未闭合则到文末）
              2) 连续≥2行的 |...| 视为一个表格原子块
              3) 占位符只含字母数字与@，不包含任何切分分隔符，绝不会在块内部被再切
        :param body: 去除标题前缀后的章节正文
        :return: (占位符掩码后的正文, {占位符: 原始块文本})
        """
        lines = body.split("\n")
        kept: List[str] = []
        blocks: Dict[str, str] = {}
        block_idx = 0
        i = 0
        while i < len(lines):
            line = lines[i]
            fence_match = _CODE_FENCE_RE.match(line)
            if fence_match:
                # 情况1：代码块起始——扫描到配对的结束围栏（同行仅有围栏字符、同种且长度不小于起始围栏）
                fence = fence_match.group(1)
                start = i
                i += 1
                while i < len(lines):
                    closed_line = lines[i].strip()
                    if closed_line and set(closed_line) == {fence[0]} and len(closed_line) >= len(fence):
                        i += 1
                        break
                    i += 1
                token = f"@@KB_ATOMIC_{block_idx}@@"
                blocks[token] = "\n".join(lines[start:i])
                kept.append(token)
                block_idx += 1
                continue

            if _TABLE_LINE_RE.match(line):
                # 情况2：表格——收集连续的表格行；仅1行的孤竖线不算表格（可能是普通文本）
                start = i
                while i < len(lines) and _TABLE_LINE_RE.match(lines[i]):
                    i += 1
                if i - start >= 2:
                    token = f"@@KB_ATOMIC_{block_idx}@@"
                    blocks[token] = "\n".join(lines[start:i])
                    kept.append(token)
                    block_idx += 1
                else:
                    kept.extend(lines[start:i])
                continue

            kept.append(line)
            i += 1

        return "\n".join(kept), blocks

    @staticmethod
    def _restore_atomic_blocks(text: str, blocks: Dict[str, str]) -> str:
        """
        【辅助函数】把占位符还原为原始代码块/表格内容
        :param text: 含占位符的Chunk文本
        :param blocks: 占位符->原文 映射
        :return: 还原后的完整文本
        """
        for token, block in blocks.items():
            if token in text:
                text = text.replace(token, block)
        return text

    def _split_long_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        【辅助函数】超长章节二次切分（核心适配LangChain分割器）
        功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
        切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
        附加保障：1.代码块/表格作为原子块整体移动，绝不从中间切断
                  2.相邻Chunk间按句子做重叠（config.overlap_sentences），缓解句子被切成两段的上下文丢失
        :param section: 原始章节字典，必须包含content键，可选title/file_title等
        :return: 切分后的子章节列表，每个子章节带父标题/序号等元信息
        """
        content = section.get("content") or ""

        # 1. 长度未超限，无需切分，直接返回
        if len(content) <= self.config.max_content_length:  # 2000
            return [section]

        # 2. 切割文本  出现了一个 超出长度的片段
        # 章节标题
        title = section.get("title") or ""
        # 计算标题的长度
        prefix = f"{title}\n\n" if title else ""  # ## 六、生物系统与物种保护\n\n

        # 计算正文的长度 2000 - 标题长度  = 去除标题 我能接受正文长度
        available_len = self.config.max_content_length - len(prefix)  # 正文的长度

        if available_len <= 0:  # 极端情况
            self.logger.warning("章节标题过长，无法切分")
            return [section]  # 标题过长 不切 直接返回

        # 去掉title部分，便于以后判断
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title):].lstrip()  # 把正文开头的标题删掉，同时清理标题后面多余空格

        # 原子块保护：先把代码块/表格换成占位符，切分器就永远无法在块内部下刀
        masked_body, atomic_blocks = self._mask_atomic_blocks(body)

        # 对正文进行切割
        # langchain的文本切割器：分隔符优先级 段落→换行→中文标点→英文标点→空格；
        # 全部用尽仍超长才硬切（如超长无标点单段），此时无法保证句子完整性，属于兜底行为
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,  # 切片长度（扣除标题）
            chunk_overlap=0,  # 重叠改由下面的句子级重叠精确控制（字符级重叠会把句子拦腰截断）
            # keep_separator='end'：langchain 1.x 默认True会把分隔符（如句号）挪到下一块开头，
            # 导致Chunk以残句开头；设为end让句号保留在前一块尾部，句子完整性与字符零丢失兼得
            keep_separator="end",
            separators=[
                "\n\n",
                "\n",
                "。",
                "？",
                "！",
                "；",
                ".",
                "?",
                "!",
                ";",
                " ",
                "",  # langchain 1.x 兜底：所有分隔符都不存在于文本时（超长无标点单段），必须显式给出""才会按chunk_size硬切
            ],
        )

        sub_sections: List[Dict[str, Any]] = []
        raw_chunks = [chunk.strip() for chunk in splitter.split_text(masked_body) if chunk.strip()]
        prev_sentences: List[str] = []  # 上一个Chunk正文的句子列表（供下一Chunk重叠取尾句）

        for idx, raw_chunk in enumerate(raw_chunks, start=1):
            # 还原代码块/表格原文，保证入库/检索的是完整块
            text = self._restore_atomic_blocks(raw_chunk, atomic_blocks)

            # 句子级重叠：本Chunk开头复制上一Chunk的最后N句（预算受限，绝不让Chunk超限）
            current_sentences = self._split_sentences(text)
            budget = available_len - len(text)  # 还原后实测剩余可用长度
            overlap_part = self._pick_overlap_sentences(prev_sentences, budget)
            if overlap_part:
                text = "\n".join(overlap_part) + "\n" + text

            # 为下一个Chunk准备尾句（基于当前Chunk正文句子）
            prev_sentences = current_sentences

            if not text:
                continue

            full_text = (prefix + text).strip()
            if len(full_text) > self.config.max_content_length:
                # 只有「原子块本身体积超过上限」时才会发生：块不可分原则优先于长度上限，记录告警供主控决策
                self.logger.warning(
                    f"Chunk长度{len(full_text)}超过上限{self.config.max_content_length}（超大代码块/表格等原子块不可切断，整体保留）"
                )
            new_title = f"{title}-{idx}"

            sub_sections.append(
                {
                    "parent_title": title,
                    "title": new_title,
                    "content": full_text,
                    "part": idx,
                    "file_title": section.get("file_title")
                }
            )

        return sub_sections

    def _step_4_refine_chunks(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
       【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
       执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
       :param sections: 步骤3处理后的章节列表
       :return: 长度适中、低碎片化的最终Chunk列表
       """
        # 阶段1：超长段落切分，所有内容由控制在最大字符数内（self.config.max_content_length）
        refined_split = []
        for section in sections:
            sub_sections = self._split_long_section(section)
            refined_split.extend(sub_sections)
        self.logger.info(f"步骤4-1：超长章节切分完成，共生成{len(refined_split)}个初始子Chunk")

        # 阶段2：短段落合并，减少内容碎片化
        final_sections = self._merge_short_sections(refined_split)
        self.logger.info(f"步骤4-2：过短章节合并完成，最终得到{len(final_sections)}个Chunk")

        # 阶段3：父标题兜底（课程要求，Milvus schema中parent_title必填）
        # 兜底规则：无parent_title则用自身title，title也无则填空字符串；同时补齐part默认值
        for section in final_sections:
            if not section.get("parent_title"):
                section["parent_title"] = section.get("title") or ""
            if "part" not in section:
                section["part"] = 1
        self.logger.debug("步骤4-3：父标题兜底完成，所有Chunk均包含parent_title字段")

        return final_sections

    def _assign_chunk_identity(self, file_title: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        【步骤4+】为每个Chunk写入稳定唯一的身份标识
        为什么：1) 切片需要在备份JSON/日志/检索链路中可追踪；
               2) Milvus入库节点用 auto_id 生成主键 chunk_id 并在插入后回填（见node_import_milvus.py），
                  切片节点若预填同名字段会与自增主键冲突导致插入失败，
                  因此这里用 chunk_index（文档内序号，1开始）+ chunk_uid（基于文件+位置哈希的确定性UUID）
        稳定性：uid = uuid5(file_title + parent_title + part + chunk_index)，同一文档重复切分得到完全相同的uid
        :param file_title: 所属文件标题
        :param sections: 最终Chunk列表
        :return: 写入身份字段后的Chunk列表（原地补充）
        """
        for idx, section in enumerate(sections, start=1):
            seed = f"{file_title}::{section.get('parent_title', '')}::{section.get('part', 1)}::{idx}"
            section["chunk_index"] = idx
            section["chunk_uid"] = str(uuid.uuid5(uuid.NAMESPACE_OID, seed))
        return sections

    def _merge_short_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        【辅助函数】过短章节合并（减少碎片化，提升检索效果）
        核心规则：仅合并「同父标题」且「当前块长度不足阈值」的相邻Chunk，且合并后不得突破max_content_length
        :param sections: 待合并的Chunk列表（通常是_split_long_section切分后的结果）
        :return: 合并后的Chunk列表
        """
        # 边界处理：空列表直接返回
        # 鲁棒性 防御性
        if not sections:
            return []

        # 合并后的结果
        merged_sections = []
        # 保存当前要合并的chunk
        current_chunk = None

        for section in sections:
            # 获取第一个section赋值给current_chunk
            if current_chunk is None:
                current_chunk = section
                continue

            # 段落是否过短
            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            is_same_parent = current_chunk.get("parent_title") == section.get("parent_title")

            # 合并前清理每个content前面重复的parent_title
            parent_title = section.get("parent_title")
            section_content = section.get("content")
            if parent_title and section_content.startswith(parent_title):
                section_content = section_content[len(parent_title):].lstrip()

            # 合并后长度守卫：短块两两相加可能突破上限（如499+1999），超限则不合并，
            # 否则下游Milvus的content字段长度约束和向量语义都会被打乱
            merged_len = len(current_chunk["content"]) + 2 + len(section_content)
            can_merge = is_current_short and is_same_parent and merged_len <= self.config.max_content_length

            if can_merge:
                # 合并段落
                current_chunk["content"] += "\n\n" + section_content

                # 处理part
                if "part" in section:
                    current_chunk["part"] = section["part"]
            else:
                # 保存当前段落
                merged_sections.append(current_chunk)
                current_chunk = section

        if current_chunk is not None:
            merged_sections.append(current_chunk)

        return merged_sections

    def _step_3_handle_no_title(self, content: str, sections: List[Dict[str, Any]], title_count: int,
                                file_title: str) -> List[Dict[str, Any]]:
        """
        【步骤3】无标题兜底处理
        功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        :param content: 标准化后的MD完整内容
        :param sections: 步骤2切分后的章节列表
        :param title_count: 步骤2识别的有效标题数量
        :param file_title: 所属文件标题
        :return: 兜底后的章节列表
        """
        if title_count == 0:
            # 将content交给LLM，总结一个标题出来
            # 提示词：根据content和file_title总结一个段落标题，不要有空行，控制在20个字符以内

            # 无标题处理
            self.logger.warning(f"步骤3：未识别到任何MD标题，将全文作为单个章节处理，文件：{file_title}")
            return [{
                "title": "无标题",
                "content": content,
                "file_title": file_title
            }]

        return sections

    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        课程v2实现：围栏支持3+个连续`/~、需与起始围栏完全配对才视为代码块结束（兼容嵌套围栏）
        :param content: 标准化后的MD完整内容（字符串）
        :param file_title: 所属文件标题，用于标记章节归属
        :return: 切分后的章节列表/有效标题数量/原始文本总行数
        """

        # 1、定义标题正则
        # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
        # \s*：行首允许0/多个空格/Tab（兼容缩进的标题）
        # #{1,6}：匹配1-6个#（对应MD1-6级标题）
        # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
        # .+：标题文字至少1个字符（避免空标题）
        title_pattern = r'\s*#{1,6}\s+.+'  # 只要是#号开头 我都认为你是标题

        # 2、初始化需要的数据
        lines = content.split("\n")  # 整个md内容 切片的列表
        # 初始化
        sections = []  # 章节列表
        title_count = 0  # 标题数量
        current_title = ""  # 当前章节的标题
        current_lines = []  # 当前标题和下一个标题之间的文本内容
        in_code_block = False  # 代码块标记：False当前没在代码块中，True当前在代码块中
        code_block_start_marker = ""  # 起始围栏原文（v2：只有同字符同长度的围栏才能关闭，避免嵌套围栏误闭合）

        # 3、定义内部函数组装sections列表
        def _flush_section():
            """内部辅助函数：将当前缓存的章节写入sections，空缓存则跳过"""
            if not current_lines:
                return
            sections.append({
                "title": current_title,  # 标题
                # 每段使用 \n换行区分
                "content": "\n".join(current_lines),  # 内容
                "file_title": file_title,  # 文件的标题
            })

        # 4、逐行遍历，识别标题和普通行以及代码块
        for line in lines:
            # 去空格
            striped_line = line.strip()

            # 4.1 识别代码块围栏边界：3个及以上连续的 ` 或 ~，起始围栏可携带语言信息串（如 ```python）
            fence_match = re.match(r"^\s*(`{3,}|~{3,})", striped_line)
            if fence_match:
                marker = fence_match.group(1)
                if not in_code_block:
                    # 进入代码块，记录起始围栏特征
                    in_code_block = True
                    code_block_start_marker = marker
                elif (
                    striped_line == marker
                    and marker[0] == code_block_start_marker[0]
                    and len(marker) >= len(code_block_start_marker)
                ):
                    # 结束围栏必须是不带信息串的纯围栏行，且与起始围栏同字符、不更短（兼容嵌套围栏）
                    in_code_block = False
                    code_block_start_marker = ""
                current_lines.append(line)  # 保留原始行（含缩进），代码块格式不能被破坏
                continue

            # 4.2 表示标题（代码块内的#号是伪标题，必须忽略）
            is_valid_title = (not in_code_block) and re.match(title_pattern, line)
            if is_valid_title:
                # 遇到标题行则先将上一个片段写入section
                _flush_section()
                current_title = striped_line  # 当前标题
                current_lines = [current_title]  # 新章节从标题行开始缓存
                title_count += 1  # 计数
                self.logger.info(f"识别标题：{current_title}")
            else:
                # 普通行：保留原始缩进（代码/表格/列表的层级信息都靠缩进）
                current_lines.append(line)

        _flush_section()
        self.logger.info(
            f"文档粗切（按标题切分）完成，共{len(sections)}个章节，标题数量是{title_count}，文本共有{len(lines)}行"
        )

        return sections, title_count, len(lines)

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        """
        【步骤1】获取并预处理输入数据
        功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
        :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
        :return: 标准化后的MD内容/文件标题
        """

        # 中间节点也可以只做轻量级的防御性判断，不做完整的磁盘级别的检查 => 前置校验 + 信任状态

        # 1、非空校验
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        md_content = state.get("md_content")  # state['md_content']
        if not md_content:
            raise StateFieldError(field_name="md_content", message="文件内容不能为空", expected_type=str)

        # 2、基础标准化：统一换行符（Windows CRLF/老Mac CR → LF），并剥离BOM头
        # BOM来自Windows记事本另存，混入后会让首个#匹配不到标题正则，导致整篇走无标题兜底
        md_content = md_content.lstrip("\ufeff")
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        return md_content, file_title


if __name__ == "__main__":
    setup_logging()

    md_path = r"D:\output\奇妙的自然界.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_path": md_path,
        "md_content": md_content,
        "file_title": "奇妙的自然界"
    }

    # 执行核心处理流程
    node_document_split = NodeDocumentSplit()
    result = node_document_split(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
