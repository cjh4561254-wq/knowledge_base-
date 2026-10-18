# processor/import_processor/nodes/node_md_img.py
"""
Markdown 图片处理节点（多模态图片理解）

本环境的链路对接（与云端 SiliconFlow/远端 MinIO 的参考实现的区别）：
- 视觉模型：本地 Ollama qwen2.5vl:3b，走 OpenAI 兼容端点（config 字段 vl_base_url /
  vl_api_key / vl_model）。qwen2.5vl:3b 是小参数量本地模型，prompt 必须短、输出格式固定，
  摘要长度受 config.img_content_length 约束；图片以 OpenAI 多模态消息格式
  {"type":"image_url","image_url":{"url":"data:image/<mime>;base64,..."}} 传入，
  这是 Ollama OpenAI 兼容层（/v1/chat/completions）官方声明的图片输入格式。
- 对象存储：统一走 utils.minio_utils（upload_file / delete_prefix / normalize_object_name），
  URL 由 minio_utils.build_public_url 生成（endpoint/scheme/前导斜杠都已归一化）。
  上传失败显式抛 ImageProcessingError —— 绝不静默返回 None，
  否则 Markdown 里会留下指向本地临时目录的死链，向量化后就是坏数据。
- 导入侧空白图过滤：pymupdf4llm 从版面切出的碎块里有一批「近乎空白/装饰性」的小图
  （实测同一份 35 页 PDF 的 85 张图里有 8 张只有 651–674 字节），它们既浪费本地
  qwen2.5vl 的识图时间，又会在检索答案下方渲染成一排小破图。本节点在「调视觉模型 +
  上传 MinIO」之前按 体积/最短边 双阈值（取或）把这类图剔除，并顺手删掉它们在
  Markdown 里的整段引用（![..](..) 与 <img src=..>），不留死链、不产生空的 ![]()。
"""
import base64
import json
import logging
import mimetypes
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

from langchain_openai import ChatOpenAI

from config.minio_config import minio_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import (
    FileProcessingError,
    ImageProcessingError,
    MinioError,
    StateFieldError,
)
from processor.import_processor.state import ImportGraphState
from utils.minio_utils import delete_prefix, normalize_object_name, upload_file

# 图片引用匹配（统一识别两种写法，忽略大小写）：
# 1) Markdown：![alt](path) / ![alt](<path> "title")，路径可带相对目录、可被 URL 编码
# 2) HTML：<img ... src="path" ...>（单双引号均可）
# 说明：mdsrc 用 [^<>\s)]+ 是因为 pymupdf4llm 产出的路径不含空格；
# 真正匹配磁盘文件时按「解码后的 basename」比对，不依赖正则本身处理编码差异。
_MD_IMG_REF_RE = re.compile(
    r"!\[[^\]]*\]\(\s*<?(?P<mdsrc>[^<>\s)>]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)"
    r"|"
    r"<img\b[^>]*?src\s*=\s*[\"'](?P<hsrc>[^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)

# 本地扩展名 → data URL 的 MIME（Ollama 会校验声明格式与字节是否一致，写错会 400）
_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}

# VL 调用彻底失败时的兜底 alt 文本（保证 MD 被替换后仍是合法图片引用，不留空串）
_FALLBACK_SUMMARY = "图片描述"


def _env_int(name: str, default: int) -> int:
    """
    读取整型环境变量，缺失/空/非法一律回退默认值。
    为什么防御式解析：一个写坏的 env 值绝不能在 import 期把整个节点模块拖崩
    （config/* 归主控统一管，本节点无权改动契约，故阈值留在节点内 + 环境变量临时覆盖）。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logging.getLogger(__name__).warning(f"环境变量{name}={raw!r}非整数，回退默认值{default}")
        return default


# 「近乎空白图」判定阈值（用户拍板：导入侧按尺寸/体积过滤，方案 D2）——取或：命中任一即判空白。
# 体积阈值拦 pymupdf4llm 版面碎块（实测空白图 651–674 字节，真截图 130KB 级，差距悬殊）；
# 边长阈值兜底拦「字节够但像素极小」的装饰块。两者均可用同名环境变量覆盖，无需改代码。
MIN_IMG_BYTES = _env_int("MIN_IMG_BYTES", 2048)
MIN_IMG_EDGE = _env_int("MIN_IMG_EDGE", 80)


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    # VL 客户端懒加载：构造 ChatOpenAI 本身不发请求，但要避免重复建 HTTP 客户端
    _vl_client: ChatOpenAI | None = None
    _vl_lock = threading.Lock()

    def process(self, state: ImportGraphState):

        """
        MD文件图片处理核心节点
        核心流程：
        1. 获取MD内容、文件路径、图片文件夹路径
        2. 扫描图片文件夹，筛选MD中实际引用的支持格式图片
        3. 调用多模态大模型为图片生成内容摘要
        4. 将图片上传至MinIO，替换MD中本地图片路径为MinIO访问URL，并填充图片摘要
        5. 备份原MD文件，保存处理后的新MD文件并更新状态

        :param state: md_path、md_content
        :return: md_path、md_content
        """

        # 步骤1：初始化数据，获取MD核心信息
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)
        if not images_dir.exists():
            self.logger.info("无图片文件夹，跳过图片处理")
            return state

        # 步骤2：扫描并筛选MD中引用的图片（同时挑出「近乎空白图」单独剔除）
        target_images, blank_images = self._step_2_scan_images(md_content, images_dir)

        # 空白图：一行汇总日志（数量 + 文件名列表），不逐张刷屏
        if blank_images:
            self.logger.info(
                f"剔除{len(blank_images)}张近乎空白/装饰碎图（不调视觉模型、不上传MinIO、"
                f"删除其Markdown引用）：{', '.join(sorted(blank_images))}"
            )

        # 既无正常图也无空白图：MD 里没有任何待处理引用，原样返回
        if not target_images and not blank_images:
            self.logger.info("未检测到MD中引用了图片，跳过图片处理")
            return state

        # 空白图文件名归一化为小写basename集合，供步骤4删除引用时比对
        blank_names = {name.lower() for name in blank_images}

        if target_images:
            # 步骤3：调用多模态大模型为「正常图」生成内容摘要
            summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)

            # 步骤4：上传正常图至MinIO，替换其MD路径为MinIO直链并填充摘要；
            #        空白图引用在同一次扫描里整段删除（不留死链、不产生空 ![]()）
            # 产出的 new_md_content 形如 ![摘要](MinIO直链)
            new_md_content = self._step_4_upload_and_replace(
                md_path_obj.stem, target_images, summaries, md_content, blank_names
            )
        else:
            # 只有空白图需要处理：不调模型、不碰MinIO，仅删除其Markdown引用
            new_md_content = self._process_md_file(md_content, {}, blank_names)

        # 步骤5：备份并保存新MD文件（原文件保留，便于回溯）
        new_md_file_name = self._step_5_backup_new_md_file(state["md_path"], new_md_content)

        # 步骤6：同步更新state —— 下游切片节点直接消费 md_content，不再回读文件；
        # md_path 也要指向新文件，保证「重新处理/断点续传」时读到的是替换后的版本
        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name

        return state

    def _step_5_backup_new_md_file(self, origin_md_path: str, md_content: str) -> str:
        """
        步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
        新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
        :param origin_md_path: 原始MD文件完整路径
        :param md_content: 处理后的新MD内容
        :return: 新MD文件的完整路径
        """

        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.logger.info(f"处理后MD文件已保存：{new_md_file_name}")
        return new_md_file_name

    def _step_4_upload_and_replace(
            self,
            doc_stem: str,
            target_images: list[tuple[str, str, tuple[str, str]]],
            summaries: dict[str, str],
            md_content: str,
            blank_names: set[str] | frozenset = frozenset()
    ) -> str:
        """
        步骤 4: 上传图片并合并信息，然后替换 Markdown 中的内容。

        流程：
        1. 确定 MinIO 上的上传目录（按文档名隔离）。
        2. 清理该目录下的旧数据（尽力而为，失败不致命：同名对象上传即覆盖）。
        3. 批量上传图片（走 minio_utils.upload_file，失败显式抛错）。
        4. 合并“图片摘要”和“图片URL”。
        5. 替换 Markdown 文本中的图片引用；blank_names 命中的空白图引用整段删除。
        :param doc_stem: 文档文件名（不含后缀），作为MinIO上传子目录名（按文档隔离）
        :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
        :param md_content: 原始MD文件内容
        :param blank_names: 被判空白的图片小写basename集合，其MD引用将被整段删除
        :return: 图片引用替换后的新MD内容
        """

        # 1. 组装上传目录：normalize_object_name 去掉 img_dir 的前导斜杠并合并重复斜杠
        #    （否则读写 key 不对称，清理永远删不掉；再去掉空格，避免 URL 里的空白字符）
        upload_dir = normalize_object_name(f"{minio_config.img_dir}/{doc_stem}").replace(" ", "")

        # 2. 清理已有目录（幂等：重新导入不堆积历史对象）
        self._clean_minio_dir(upload_dir)

        # 3. 批量上传（任一图片上传失败会抛 ImageProcessingError，节点显式失败）
        urls = self._upload_images_batch(upload_dir, target_images)

        # 4. 合并图片摘要和URL
        image_info = self._merge_summary_and_url(summaries, urls)

        # 5. 替换MD中的图片引用（正常图 → MinIO直链+摘要；空白图 → 整段删除）
        md_content = self._process_md_file(md_content, image_info, blank_names)

        return md_content

    def _process_md_file(
            self,
            md_content: str,
            image_info: dict[str, tuple[str, str]],
            blank_names: set[str] | frozenset = frozenset()
    ) -> str:
        """
        核心功能：把MD中的本地图片引用改写为MinIO远程引用，并删除「近乎空白图」的整段引用。
        替换规则：![原描述](本地路径) / <img src="本地路径"> → ![图片摘要](MinIO访问URL)
        空白规则：命中 blank_names 的引用 ![..](..) / <img src=..> → 整段删除（返回空串），
                 天然不留死链、不留空的 ![]()（删的是整段匹配，不残留括号骨架）。

        为什么用「单次全局扫描 + 回调判定」而不是逐文件 re.sub(.*name.*)：
        - 一次遍历同时覆盖 Markdown 与 HTML <img> 两种引用写法、大小写差异、
          相对/绝对路径、URL 编码过的路径（解码后按 basename 比对）；
        - 旧的 `.*?图片名.*?` 会把 "xa.png" 误判成包含 "a.png" 的引用，且无法
          跨 `)` 边界收束，回调方式按整段引用替换，天然无此问题。
        :param md_content: 原始MD文件内容
        :param image_info: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)
        :param blank_names: 被判空白的图片小写basename集合，其引用整段删除
        :return: 处理后的新MD内容
        """

        # 磁盘文件名（小写）→ (摘要, URL)，实现大小写不敏感匹配
        lookup = {name.lower(): (summary, url) for name, (summary, url) in image_info.items()}
        replaced = 0
        removed = 0

        def _repl(match: re.Match) -> str:
            nonlocal replaced, removed
            raw = match.group("mdsrc") or match.group("hsrc") or ""
            base = self._ref_basename(raw).lower()
            # 先判空白图：命中即整段删除（返回空串，删的是完整引用段，不会残留空骨架）
            if base in blank_names:
                removed += 1
                return ""
            hit = lookup.get(base)
            if hit is None:
                # 该引用指向磁盘上不存在/未通过校验的文件：保持原样，由调用日志提示
                return match.group(0)
            summary, new_url = hit
            replaced += 1
            # 摘要先净化过（无换行/方括号），URL 由 minio_utils 生成（无空格）
            return f"![{summary}]({new_url})"

        new_content = _MD_IMG_REF_RE.sub(_repl, md_content)
        if removed:
            self.logger.info(f"已删除{removed}处空白图片的Markdown引用")
        self.logger.info(f"MD文件图片引用替换完成，共替换{replaced}处（待替换图片{len(lookup)}张）")

        return new_content

    def _merge_summary_and_url(self, summaries: dict[str, str], urls: dict[str, str]) -> dict[str, tuple[str, str]]:
        """
        合并图片摘要字典和URL字典，过滤掉无URL的图片（防御式：正常情况下上传失败已抛错，
        这里只是保证字典结构完整，绝不把 None 拼进 Markdown）
        :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
        :param urls: 图片URL字典，键：图片文件名，值：MinIO访问URL
        :return: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)元组
        """
        image_info = {}
        for image_file, summary in summaries.items():
            if url := urls.get(image_file):  # 海象运算符
                image_info[image_file] = (summary, url)
        return image_info

    def _upload_images_batch(self, upload_dir: str, target_images: list[tuple]) -> dict[str, str]:
        """
        批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
        :param upload_dir: MinIO上传根目录（已归一化的对象前缀）
        :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :return: 图片URL字典，键：图片文件名，值：MinIO访问URL
        :raises ImageProcessingError: 任一图片上传失败（文件缺失/存储不可用）。
                为什么必须抛而不是记 None：返回 None 会让该图片在 Markdown 里
                保留本地临时路径的死链，并且流程「看似成功」地继续向量化。
        """
        urls = {}
        for img_file, img_path, _ in target_images:
            # 文件名里的空格从对象名中剔除：build_public_url 不做百分号编码，
            # 带空格的直链写进 Markdown 会被渲染端截断（pymupdf4llm 产出的是哈希名，
            # 这里主要防御外部/手工 MD）
            safe_name = img_file.replace(" ", "")
            object_name = f"{upload_dir}/{safe_name}"
            try:
                # upload_file 内部走 build_public_url：scheme/endpoint/key 归一化统一处理，
                # 不再手工拼 f"http://{endpoint}/..."（endpoint 配了 https 或被加路径时会拼错）
                urls[img_file] = upload_file(img_path, object_name, node_name=self.name)
            except MinioError as e:
                raise ImageProcessingError(
                    message=f"图片上传MinIO失败，终止流程以避免Markdown留下本地死链：{img_file}",
                    node_name=self.name,
                    cause=e,
                ) from e
        return urls

    def _clean_minio_dir(self, upload_dir: str) -> None:
        """
        幂等清理：上传前删除 MinIO 中该文档目录下的旧对象。
        失败仅告警不抛错——同名对象上传本身会覆盖，清理只是防止改名/删除后残留。
        :param upload_dir: MinIO目录前缀（已归一化，无前导斜杠）
        """
        try:
            deleted = delete_prefix(upload_dir, node_name=self.name)
            if deleted:
                self.logger.info(f"已清理MinIO旧图片目录 {upload_dir}，共{deleted}个对象")
        except MinioError as e:
            self.logger.warning(f"清理MinIO旧目录失败（不影响本次上传，同名对象将被覆盖）：{e}")

    def _step_3_generate_summaries(self, doc_stem: str,
                                   target_images: list[tuple[str, str, tuple[str, str]]]) -> dict[str, str]:
        """
        步骤3：线程池并发为图片生成内容摘要，用滑动窗口限流约束VL调用速率。

        并发模型（为什么这样设计）：
        - 提交循环在**主线程**里过限流器，deque 单线程访问、无锁、不可能死等；
        - 工作线程只做「发一次请求、拿结果」，不阻塞在限流器上，
          彻底避免旧实现里「N 个线程排队 sleep 同一窗口」的踩踏与饥饿；
        - 实际请求速率由提交侧的 requests_per_minute 约束（Ollama 本地模型按此节流即可）。
        :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
        :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
        """

        summaries: dict[str, str] = {}
        if not target_images:
            return summaries

        # 1. 速率参数：每分钟最多 requests_per_minute 次（当前配置 15）
        max_requests = max(1, int(self.config.requests_per_minute))
        window_seconds = 60.0

        # 2. 线程数受配置约束：本地 qwen2.5vl:3b 吞吐有限，取 RPM 的 1/4 档并封顶 4，
        #    并发过高只会让 Ollama 排队，无收益
        max_workers = max(1, min(4, max_requests // 4))

        # 3. 滑动窗口时间戳队列（仅主线程读写）
        request_deque: deque[float] = deque()

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vl_summary") as pool:
            futures = {}
            for img_file, img_path, context in target_images:
                # 3.1 限速：达到窗口上限时等待，最长等待被封顶，绝不无限等
                self._apply_api_rate_limit(request_deque, max_requests, window_seconds)

                # 3.2 提交摘要任务（_summarize_image 内部自带降级，不抛异常）
                futures[pool.submit(self._summarize_image, img_path, doc_stem, context)] = img_file

            for future, img_file in futures.items():
                try:
                    summaries[img_file] = future.result()
                except Exception as e:  # noqa: BLE001 —— 单图失败必须降级，不许拖垮整节点
                    # 防御兜底：单张图片失败不拖垮整节点
                    self.logger.warning(f"图片摘要任务异常，使用兜底文案：{img_file}，原因：{e}")
                    summaries[img_file] = _FALLBACK_SUMMARY

        return summaries

    def _get_vl_model(self) -> ChatOpenAI:
        """
        懒加载视觉模型客户端（Ollama 的 OpenAI 兼容端点）。

        为什么不用文本 LLM 的 base_url/api_key：VL 走的是本机 Ollama
        （vl_base_url=http://localhost:11434/v1、vl_api_key 为占位 key），
        与 DeepSeek 文本通道完全分开，误用云端 key 会 401。
        """
        if self._vl_client is None:
            with self._vl_lock:
                if self._vl_client is None:
                    self._vl_client = ChatOpenAI(
                        model=self.config.vl_model,
                        # Ollama 不校验 key，但 OpenAI SDK 要求非空，占位 "ollama"
                        api_key=self.config.vl_api_key or "ollama",
                        base_url=self.config.vl_base_url,
                        # 本地小模型：temperature=0 让输出格式更稳定（固定一句话摘要）
                        temperature=0,
                        # 单次请求超时：防止某张图卡死导致线程池死等；重试 1 次后走兜底文案
                        timeout=120,
                        max_retries=1,
                    )
        return self._vl_client

    def _summarize_image(self, image_path: str, root_folder: str, image_content: tuple[str, str]) -> str:
        """
        调用本地 Ollama 多模态模型（qwen2.5vl:3b）总结图片内容。

        消息格式依据：Ollama OpenAI 兼容层 /v1/chat/completions 的多模态消息，
        图片以 {"type":"image_url","image_url":{"url":"data:image/<mime>;base64,..."}}
        形式传入（与 OpenAI Chat Completions 图片输入格式一致）。
        注意：data URL 声明的 MIME 必须与真实字节一致，所以按扩展名映射，
        不能像旧实现那样一律写 image/jpeg（png 会被 Ollama 解码报错）。

        参数：
        - image_path: 图片本地路径。
        - root_folder: 文档所属文件夹名（提供更多上下文）。
        - image_content: 图片在文档中的上下文 (前文, 后文)。
        :return: 中文摘要（已截断到 img_content_length 字符，异常时返回兜底文案）
        """

        # 1. 读取图片字节并转 base64（损坏/丢失的图片在步骤2已被过滤，这里是防御式兜底）
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        except OSError as e:
            self.logger.warning(f"图片不可读，跳过VL总结：{image_path}，原因：{e}")
            return _FALLBACK_SUMMARY

        # 2. 按扩展名确定真实 MIME
        ext = os.path.splitext(image_path)[1].lower()
        mime = _MIME_BY_EXT.get(ext) or mimetypes.guess_type(image_path)[0] or "image/png"

        # 3. 摘要长度上限（本地小模型对长指令遵循差，prompt 保持短而固定）
        max_len = max(20, int(self.config.img_content_length))

        # 上下文各截 40 字：给模型定位用，太长会稀释小模型的注意力
        pre_hint = (image_content[0] or "")[-40:]
        post_hint = (image_content[1] or "")[:40]

        try:
            chat_model = self._get_vl_model()

            prompt = (
                f"这是《{root_folder}》文档中的一张图片，"
                f"图片前文「{pre_hint}」后文「{post_hint}」。"
                f"请用中文一句话概括图片内容，不超过{max_len}个字，只输出概括本身。"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                # Ollama /v1 兼容层要求标准 data URL 形式
                                "url": f"data:{mime};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

            response = chat_model.invoke(messages)
            content = response.content
            if isinstance(content, list):
                # langchain 多模态响应可能返回分段列表，统一拼成纯文本
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part) for part in content
                )
            return self._sanitize_summary(str(content), max_len)

        except Exception as e:  # noqa: BLE001 —— VL调用异常走兜底文案（降级路径）
            # 降级路径：单图失败不抛错（上传与替换仍要继续），用兜底文案
            self.logger.warning(f"获取图片摘要失败：{image_path}，错误：{e}")
            return _FALLBACK_SUMMARY

    @staticmethod
    def _sanitize_summary(text: str, max_len: int) -> str:
        """
        净化摘要文本：去掉换行与方括号/括号（避免破坏 ![..](..) 的 Markdown 结构），
        截断到 img_content_length，空结果回退兜底文案
        """
        cleaned = re.sub(r"\s+", "", text or "").translate(str.maketrans("", "", "[]()「」"))
        cleaned = cleaned[:max_len].strip()
        return cleaned or _FALLBACK_SUMMARY

    def _apply_api_rate_limit(
            self,
            request_times: deque[float],
            max_requests: int,
            window_seconds: int = 60
    ) -> None:
        """
        通用滑动窗口API速率限制器
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则等待最早请求滑出窗口。

        与旧实现的两点区别：
        1. 等待后**重新检查**（while 循环），旧版睡一觉就直接放行，窗口没滑出来会超发；
        2. 总等待时长封顶（window_seconds + 5s）：时间戳队列只要单调就必然在
           一个窗口内排空，加封顶是防御时钟回拨/系统挂起等病态场景，绝不无限死等。
        :param request_times: 存储请求时间戳的双端队列，由调用方（提交线程）独占
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待（有上限）
        """
        # 死等保护：本函数最长阻塞 一个窗口 + 5 秒
        deadline = time.monotonic() + window_seconds + 5.0

        while True:
            current_time = time.time()

            # 1. 清理滑动窗口中的过期请求
            while request_times and current_time - request_times[0] >= window_seconds:
                request_times.popleft()

            # 2. 窗口内未达上限：直接放行
            if len(request_times) < max_requests:
                break

            # 3. 达到上限：等待最早请求滑出窗口（带总时长封顶，绝不无限等）
            sleep_duration = window_seconds - (current_time - request_times[0])
            if time.monotonic() >= deadline:
                self.logger.warning(
                    f"限流等待已达上限({window_seconds + 5:.0f}s)，直接放行以避免死等"
                )
                break
            if sleep_duration > 0:
                self.logger.info(f"请求被限速（{window_seconds:.0f}s内已{len(request_times)}次），等待{sleep_duration:.2f}秒...")
                time.sleep(sleep_duration)

        # 4. 记录当前请求的时间戳，新请求入队
        request_times.append(time.time())
        self.logger.debug(f"{self.name} 请求放行，当前{window_seconds:.0f}s窗口内请求次数为{len(request_times)}")

    def _step_2_scan_images(
            self, md_content: str, images_dir: Path
    ) -> tuple[list[tuple[str, str, tuple[str, str]]], list[str]]:
        """
        扫描图片文件夹与MD引用，产出「支持格式 + 文件完好 + 非空白 + MD中实际引用」的图片清单，
        并单独收集被判定为「近乎空白」的文件名（供上层删除其MD引用）。

        匹配健壮性（对照旧实现的「遍历磁盘文件 + 模糊正则搜文件名」）：
        - 统一用 _MD_IMG_REF_RE 扫描 MD：同时支持 ![alt](path) 与 <img src=path>；
        - 引用路径先做 URL 解码（unquote）再取 basename，大小写不敏感比对磁盘文件名，
          相对路径（images/x.png）、子目录路径、Windows 反斜杠路径都能命中；
        - 磁盘侧先做扩展名过滤与 PIL 完好性校验，损坏/空文件/目录直接 WARNING 跳过，
          不会让整个节点崩；
        - 完好之后再做「空白图」判定（体积/最短边取或），命中的图不进待处理清单，
          也就不会消耗视觉模型时间与MinIO存储。
        :param md_content: MD文件完整内容
        :param images_dir: 图片文件夹路径对象
        :return: 元组(待处理图片列表[(文件名, 完整路径, 上下文)], 空白图文件名列表)
        """

        # 1. 收集磁盘上合法、完好且非空白的图片：小写basename → 磁盘上的真实文件名；
        #    blank_files 单独收集被判空白的原始文件名（上层据此删引用 + 汇总日志）
        disk_names: dict[str, str] = {}
        blank_files: list[str] = []
        for image_file in os.listdir(images_dir):
            img_path = images_dir / image_file
            if not img_path.is_file():
                # 子目录等非文件条目：跳过，os.listdir 会一并返回
                continue

            # 1.1 过滤不支持的扩展名（非图片文件：明确降级，继续处理其它图片）
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in self.config.image_extensions:
                self.logger.warning(f"图片{image_file}格式不支持，跳过")
                continue

            # 1.2 完好性校验：空文件/损坏文件 WARNING 跳过（PIL 缺失时退化为大小校验）
            #     注意：0 字节文件在这里已被拦掉，不会走到下面的空白判定
            if not self._is_readable_image(str(img_path)):
                continue

            # 1.3 空白/装饰碎图过滤（用户拍板 D2：导入侧按尺寸/体积过滤）
            #     在调视觉模型、上传MinIO之前拦下：既省识别时间，又防答案里出现小破图
            if self._is_blank_image(str(img_path)):
                blank_files.append(image_file)
                continue

            disk_names[image_file.lower()] = image_file

        # 2. 扫描MD中的所有图片引用，命中磁盘的取首个匹配位置作为上下文
        contexts: dict[str, tuple[str, str]] = {}
        for match in _MD_IMG_REF_RE.finditer(md_content):
            raw = match.group("mdsrc") or match.group("hsrc") or ""
            hit = disk_names.get(self._ref_basename(raw).lower())
            if hit is None:
                continue
            if hit not in contexts:
                contexts[hit] = self._extract_context(md_content, match.start(), match.end())

        # 3. 磁盘有但MD未引用的图片：提示后跳过（旧行为保留）
        for missing in set(disk_names.values()) - set(contexts):
            self.logger.warning(f"图片{missing}未在md文档中被引用，跳过")

        target_images = [(name, str(images_dir / name), contexts[name]) for name in contexts]
        return target_images, blank_files

    def _is_blank_image(self, img_path: str) -> bool:
        """
        判定「近乎空白的图」：文件字节数 < MIN_IMG_BYTES 或 最短边 < MIN_IMG_EDGE（取或）。

        判定顺序（先便宜后贵）：
        - 先 os.path.getsize：pymupdf4llm 版面碎块典型仅几百字节，读一次 stat 就能拦下，
          省掉对这批图做 PIL 尺寸解析（更省掉后面 944s 识图里废图那部分的时间）；
        - 再用 PIL 读一次尺寸兜底拦「字节够但像素极小」的装饰块（本机 CPU 上 open+读 size
          很快，且不解码全图）。

        放行优先（宁漏不错杀，避免误删真实截图）：
        - getsize 失败（并发删除等）：不当空白，交回原有完好性/上传链路；
        - PIL 读不出尺寸（损坏、或 gif 只读首帧仍拿不到有效尺寸）：当非空白放行，
          绝不误删。
        :param img_path: 图片磁盘完整路径（可含中文/空格，os/PIL 在 Windows 下均按 str 处理）
        :return: True 表示应被剔除的空白图
        """
        # 1. 体积阈值（最便宜，先判）
        try:
            size_bytes = os.path.getsize(img_path)
        except OSError:
            return False
        if size_bytes < MIN_IMG_BYTES:
            return True

        # 2. 尺寸阈值：取最短边；读不出尺寸（含 gif 首帧异常）一律放行
        dims = self._image_dims(img_path)
        if dims is None:
            return False
        width, height = dims
        return min(width, height) < MIN_IMG_EDGE

    @staticmethod
    def _image_dims(img_path: str) -> tuple[int, int] | None:
        """
        用 PIL 读图片像素尺寸（(宽, 高)）。只 open + 读 .size，不解码全图，CPU 上一次开销很小。
        对 gif：PIL 打开返回的是第一帧的尺寸，符合「只读第一帧」要求。
        任何读不出有效正尺寸的情况（损坏/格式异常/0 尺寸）都返回 None，交调用方按「非空白」放行。
        """
        try:
            from PIL import Image
        except ImportError:
            # 无 Pillow 环境：无法判尺寸，一律放行（与 _is_readable_image 的退化策略一致）
            return None
        try:
            with Image.open(img_path) as im:
                width, height = im.size
            if width > 0 and height > 0:
                return int(width), int(height)
            return None
        except Exception:  # noqa: BLE001 —— 读尺寸失败绝不等于空白，交回原链路，不误删
            return None

    @staticmethod
    def _ref_basename(ref_path: str) -> str:
        """
        归一化MD引用路径用于比对：URL解码 → file:// 剥离 → 反斜杠转正斜杠 → 取basename。
        （pymupdf4llm 产出 images/<hash>.png；其它来源可能是 ./images/、绝对路径、
        %20编码过的路径，统一解码后只按最后一段文件名匹配磁盘文件）
        """
        decoded = unquote(ref_path or "").strip()
        if decoded.lower().startswith("file://"):
            decoded = decoded[len("file://"):].lstrip("/")
        decoded = decoded.replace("\\", "/")
        return decoded.rsplit("/", 1)[-1]

    @staticmethod
    def _extract_context(md_content: str, start: int, end: int, context_len: int = 100) -> tuple[str, str]:
        """截取图片引用位置前后各 context_len 字符作为VL上下文（防索引越界）"""
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        return pre_text, post_text

    def _is_readable_image(self, img_path: str) -> bool:
        """
        图片完好性校验：能被 PIL 打开且 verify() 通过才算完好。
        校验失败只 WARNING 并跳过该图（降级路径），不抛异常、不中断节点。
        PIL 不可用时退化为「存在且非空」校验，保证节点在无 Pillow 环境也能跑。
        """
        try:
            if os.path.getsize(img_path) == 0:
                self.logger.warning(f"图片为空文件，跳过：{img_path}")
                return False
        except OSError as e:
            self.logger.warning(f"图片不可访问，跳过：{img_path}，原因：{e}")
            return False

        try:
            from PIL import Image
        except ImportError:
            return True

        try:
            with Image.open(img_path) as im:
                im.verify()  # 只校验头信息与完整性，不解码全图（快）
            return True
        except Exception as e:  # noqa: BLE001 —— 损坏图片只跳过，不中断节点
            self.logger.warning(f"图片已损坏或格式非法，跳过：{img_path}，原因：{e}")
            return False

    def _step_1_get_content(self, state: ImportGraphState) -> tuple[str, Path, Path]:
        """
        从全局状态中提取并初始化MD处理所需核心数据
        :param state: 流程全局状态对象
        :return: 元组(MD文件内容, MD文件路径, 图片文件夹路径)
        :raise FileProcessingError: 当状态中无有效MD文件路径时抛出
        """

        # 1. 参数非空校验
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(
                field_name="md_path",
                message="MD文件路径不能为空",
                expected_type=str)

        # 2. 路径转换
        md_path_obj = Path(md_path)

        # 3. 检查文件的有效性
        if not md_path_obj.exists():
            raise FileProcessingError(message=f"文件{md_path_obj.name}不存在")

        # 4. 获取md_content（以磁盘文件为准：上一节点已写入 {file_dir}/{stem}/{stem}.md，
        #    state 中的 md_content 可能是空串）
        md_content = md_path_obj.read_text(encoding="utf-8")

        # 5. 获取md文件的图片文件夹路径
        #    约定：PDF转MD节点产出 {file_dir}/{stem}/{stem}.md 与 {file_dir}/{stem}/images/*.png
        img_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, img_dir


if __name__ == "__main__":
    setup_logging()

    # 手动调试入口：python node_md_img.py <md文件路径>
    import sys

    test_md_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\output\hak180产品安全手册\hak180产品安全手册.md"
    if not os.path.exists(test_md_path):
        logging.getLogger().error(f"MD文件不存在：{test_md_path}")
        raise SystemExit(1)

    init_state = {
        "task_id": "debug-node-md-img",
        "md_path": test_md_path,
        "md_content": "",
    }

    # 执行核心处理流程
    node_md_img = NodeMDImg()
    result = node_md_img(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
