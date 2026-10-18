# processor/import_processor/nodes/node_pdf_to_md.py
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path

import requests

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import (
    ConfigurationError,
    FileProcessingError,
    PdfConversionError,
    StateFieldError,
)
from processor.import_processor.state import ImportGraphState

# MinerU 轮询参数：为什么写成常量——在线解析大文件很慢，必须有上限，
# 否则 token 失效/服务异常时节点会永远卡在轮询里，把整条导入流水线拖死。
MINERU_POLL_TIMEOUT_SECONDS = 600  # 轮询总超时
MINERU_POLL_INTERVAL_SECONDS = 3   # 轮询间隔
MINERU_HTTP_TIMEOUT_SECONDS = 30   # 单次 HTTP 请求超时（防止 requests 本身挂起）


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析

    支持两种解析引擎，由配置项 pdf_engine 切换：
    - pymupdf4llm（默认）：本地解析，无外部依赖、免费、离线可用
    - mineru：在线 API 解析，识别精度更高，但需要有效 MINERU_API_TOKEN
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        """
        主流程：校验路径 → 按引擎开关转换 PDF 得到 MD 路径 → 读取 MD 内容 → 写回状态

        :param state: 输入 `pdf_path`、`file_dir`
        :return: 更新 `md_path`、`md_content`
        """

        # 1. 参数校验并返回Path结果
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 2. 引擎开关：优先读 config_base 的 pdf_engine，兜底读 mineru_config，最后默认本地引擎
        engine = (getattr(self.config, "pdf_engine", "") or getattr(mineru_config, "pdf_engine", "")
                  or "pymupdf4llm").strip().lower()

        if engine == "mineru":
            # 远程分支：上传MinerU并轮询结果，下载ZIP解压得到md绝对路径
            zip_url = self._step_2_upload_and_poll(pdf_path_obj)
            self.logger.info(zip_url)
            md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)
        else:
            if engine != "pymupdf4llm":
                self.logger.warning(f"未知的 pdf_engine 配置：{engine}，回退使用本地 pymupdf4llm 引擎")
            # 本地分支：pymupdf4llm 直接转换，产出目录结构与 MinerU 解压目录保持一致
            md_path = self._step_2_convert_with_pymupdf4llm(pdf_path_obj, output_dir_obj)

        # 3. 将MD文件的内容都取出来
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 4. 返回结果（state 字段名与 state.py 定义保持一致，下游节点依赖）
        state["md_content"] = md_content
        state["md_path"] = md_path
        return state

    def _step_2_convert_with_pymupdf4llm(self, pdf_path_obj: Path, output_dir_obj: Path) -> str:
        """
        步骤2（本地引擎分支）：使用 pymupdf4llm 将 PDF 转换为 Markdown

        核心流程：清理/创建输出目录 → 转 MD 并抽取图片 → 写出 MD 文件
        目录约定（与 MinerU 分支及下游 node_md_img 完全一致）：
            {file_dir}/{pdf_stem}/{pdf_stem}.md          —— MD 文件
            {file_dir}/{pdf_stem}/images/xxx.png         —— 图片
        下游 node_md_img 用 `md_path.parent / "images"` 定位图片目录，
        并用宽松正则按图片文件名在 MD 中匹配引用，故 MD 内的图片引用写成
        绝对路径（正斜杠）也可被正确识别。
        参数：pdf_path_obj-已校验的PDF Path对象；output_dir_obj-输出根目录Path
        返回：最终MD文件的字符串格式绝对路径
        异常：PdfConversionError(转换失败或产出空内容)
        """

        # 惰性导入：pymupdf 是本机 C 扩展，导入很快，但只有走本地分支才需要，
        # 避免 mineru 分支/其他流程被迫依赖该包
        import pymupdf4llm

        pdf_stem = pdf_path_obj.stem
        target_dir = output_dir_obj / pdf_stem

        # 删除已有目录，保证重复导入同一文件时不残留旧图片（与 MinerU 分支行为一致）
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 图片输出目录：必须叫 images，且与 MD 文件同级（下游节点硬约定）
        images_dir = target_dir / "images"
        # pymupdf4llm 在真正有图片落盘时才会创建该目录，无图片则目录不存在，node_md_img 会自行跳过

        self.log_step("本地转换", f"pymupdf4llm 开始解析：{pdf_path_obj}")
        try:
            md_text = pymupdf4llm.to_markdown(
                str(pdf_path_obj),        # 传入文档路径（Windows 反斜杠路径 pathlib/os 均可处理）
                write_images=True,        # 抽取页面图片到本地文件
                image_path=str(images_dir),  # 图片目录：{file_dir}/{stem}/images
                image_format="png",       # png 在 config.image_extensions 白名单内且无损
                page_chunks=False,        # 返回整篇 MD 字符串
            )
        except Exception as e:  # noqa: BLE001 —— pymupdf 对损坏/加密PDF会抛多种底层异常，统一转业务异常
            raise PdfConversionError(f"pymupdf4llm 解析失败：{pdf_path_obj}，错误信息：{e}")

        if not md_text or not md_text.strip():
            raise PdfConversionError(f"pymupdf4llm 转换结果为空：{pdf_path_obj}，可能是加密PDF或损坏文件")

        # 写出 MD 文件：文件名统一为 PDF 同名（与 MinerU 分支的 full.md 重命名规则一致）
        md_file = target_dir / f"{pdf_stem}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_text)

        image_count = len(os.listdir(images_dir)) if images_dir.exists() else 0
        self.log_step("本地转换", f"完成：{md_file}，共抽取图片 {image_count} 张，MD 内容长度 {len(md_text)} 字符")

        return str(md_file.absolute())

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        """
        步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
        核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
        参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
        返回：最终MD文件的字符串格式绝对路径
        异常：FileProcessingError(下载失败/缺少MD文件)
        """

        # 1.下载zip  包含压缩包 服务器返回的内容（带超时，防止下载挂起）
        response = requests.get(zip_url, timeout=MINERU_HTTP_TIMEOUT_SECONDS)

        # 防御性编程  （目的是为了代码的鲁棒性（健壮性））
        if response.status_code != 200:
            raise FileProcessingError(f"下载失败，状态码：{response.status_code}")

        #定义做包的保存路径
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            f.write(response.content)

        self.log_step("ZIP下载", "下载成功")# 打印

        # 2. 解压ZIP
        # 解压目录
        extract_target_dir = output_dir_obj / pdf_stem

        # 删除已有目录  防御性编程
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)

        # 创建新目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 解压ZIP
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)

        self.log_step("ZIP下载", "解压完成")
        # 读extract_target_dir full.md
        target_md_file = extract_target_dir / "full.md"# 路径拼接
        # 校验MD文件存在：ZIP结构变化时给出明确错误，而不是 rename 抛裸 OSError
        if not target_md_file.exists():
            raise FileProcessingError(f"MinerU结果ZIP包中未找到 full.md 文件，解压目录：{extract_target_dir}")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")# 新的md的名字
        target_md_file.rename(new_md_path)# 执行

        return str(new_md_path.absolute())# 绝对路径

    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ConfigurationError(配置缺失)、PdfConversionError(请求/上传/认证失败)、
             TimeoutError(任务超时)
        """

        # 1、配置文件校验：token 缺失/无效必须在这里快速失败，绝不能进入轮询死等
        if not self.config.mineru_base_url:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_BASE_URL 参数")
        if not self.config.mineru_api_token:
            raise ConfigurationError(
                "MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 参数；"
                "若暂无有效 token，请设置 PDF_TO_MD_ENGINE=pymupdf4llm 使用本地解析引擎")

        # 2. 调用MinerU的远程API接口，获取上传链接
        # 2.1 组织上传需要的数据
        token = self.config.mineru_api_token
        url = f"{self.config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }

        # 2.2 发送请求（带超时，防止 requests 挂起）
        try:
            response = requests.post(url, headers=header, json=data, timeout=MINERU_HTTP_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            raise PdfConversionError(f'请求MinerU上传链接异常：{e}')

        # 2.3 获取响应结果
        if response.status_code != 200:
            raise PdfConversionError(
                f'获取上传链接失败. 响应状态:{response.status_code} ,响应结果:{response}')

        try:
            result = response.json()
        except ValueError:
            raise PdfConversionError(f'获取上传链接失败. MinerU返回非JSON响应：{response.text[:200]}')

        # 业务码非0：包含认证失败（如 A0202 token 无效），立即报错并给出切回本地引擎的提示
        if result.get("code") != 0:
            raise PdfConversionError(
                '获取上传链接失败(MinerU业务码:{}，信息:{})。'
                '常见原因为 token 无效/余额不足，可在 .env 设置 PDF_TO_MD_ENGINE=pymupdf4llm 改用本地引擎'
                    .format(result.get("code"), result.get("msg", result)))

        # 2.4 获取到上传链接（校验响应结构，避免 KeyError 裸崩）
        data_block = result.get("data") or {}
        batch_id = data_block.get("batch_id")
        urls = data_block.get("file_urls")
        if not batch_id or not urls:
            raise PdfConversionError(f'获取上传链接失败. 响应数据缺少 batch_id/file_urls：{result}')

        # 3. 文件上传
        with open(pdf_path_obj, 'rb') as f:
            res_upload = requests.put(urls[0], data=f, timeout=MINERU_HTTP_TIMEOUT_SECONDS * 2)
            if res_upload.status_code != 200:
                raise PdfConversionError(f'{urls[0]}文件上传失败，状态码：{res_upload.status_code}')

            self.logger.info('上传文件成功!')

        # 4. 获取解析结果
        poll_url = f"{self.config.mineru_base_url}/extract-results/batch/{batch_id}"

        # 轮询获取：有硬性总超时，任何异常路径下都不会无限循环
        start_time = time.time()  # 记录当前时间
        timeout_seconds = MINERU_POLL_TIMEOUT_SECONDS  # 最大超时时间
        poll_interval = MINERU_POLL_INTERVAL_SECONDS  # 轮询间隔
        self.log_step(
            step_name="轮询开始",
            message=f"轮询间隔: {poll_interval}s, 超时时间: {timeout_seconds}s，batch_id:{batch_id}"
        )
        while True:
            #已消耗时间
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
              raise TimeoutError(f"[任务轮询]超时，已消耗时间: {int(elapsed_time)}s，batch_id:{batch_id}")

            # 获取任务结果
            try:
                poll_res = requests.get(poll_url, headers=header, timeout=10)
            except Exception as e:  # noqa: BLE001 —— 轮询中任何网络异常都按重试处理，总超时兜底
                self.logger.warning(f"网络请求异常，{poll_interval}s后重试，batch_id:{batch_id}")
                self.logger.warning(f"异常信息：{e}")
                time.sleep(poll_interval)
                continue

            if poll_res.status_code != 200:
                raise PdfConversionError(f'[任务轮询]失败，状态码：{poll_res.status_code}')

            poll_res_json = poll_res.json()
            if poll_res_json.get("code") != 0:
                raise PdfConversionError(
                    f'[任务轮询]失败，错误码：{poll_res_json.get("code")}，信息：{poll_res_json.get("msg", "")}')

            extract_results = (poll_res_json.get("data") or {}).get("extract_result")
            if not extract_results:
                raise PdfConversionError(f'[任务轮询]失败，响应缺少 extract_result：{poll_res_json}')
            extract_result = extract_results[0]
            # 获取任务的状态值
            data_state = extract_result.get("state")
            if data_state == "done":
                self.log_step(
                    step_name="任务轮询",
                    message=f"解析完成s, 总耗时: {int(elapsed_time)}s，batch_id:{batch_id}"
                )

                full_zip_url = extract_result.get("full_zip_url")
                if not full_zip_url:
                    raise PdfConversionError(f'[任务轮询]任务done但未返回zip下载地址，batch_id:{batch_id}')
                self.log_step(
                    step_name="任务轮询",
                    message=f"获取全量zip地址成功: {full_zip_url}, 总耗时: {int(elapsed_time)}s，batch_id:{batch_id}"
                )

                return full_zip_url

            elif data_state == "failed":
                err_msg = extract_result.get("err_msg", "未知错误，无具体信息")
                raise PdfConversionError(f'[任务轮询]失败，错误信息: {err_msg}')

            else:
                self.log_step(
                    step_name="任务轮询",
                    message=f"处理中...... 已耗时: {int(elapsed_time)}s，状态：{data_state}，batch_id:{batch_id}"
                )
                time.sleep(poll_interval)

    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：StateFieldError(参数缺失)、FileProcessingError(文件无效)
        """

        # 1.参数的非空校验
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", message="PDF路径不能为空", expected_type=str)

        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", message="输出路径不能为空", expected_type=str)

        # 2. 转换为Path对象（pathlib 统一处理 Windows 反斜杠 / 盘符 / Git Bash 风格路径）
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        # 3.pdf是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(f"文件{pdf_path_obj.name}不存在")

        # 4.输出目录不存在则创建
        if not file_dir_obj.exists():
            self.logger.info(f"输出目录{file_dir_obj.absolute()}不存在，开始创建...")
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_dir_obj


if __name__ == "__main__":
    # 激活日志
    setup_logging()

    init_state = {
        "task_id": "node_pdf_to_md_demo",
        "pdf_path": r"D:\file\20261010\bbdd6d88-6de9-42ec-ae08-b4ce42d1d6c5\sample.pdf",
        "file_dir": os.getenv("MD_ROOT_DIR", "./temp-files/")
    }
    node_pdf_to_md = NodePDFToMD()
    # 使用 对象() 的方式相当于调用了 对象的__call__()
    result = node_pdf_to_md(init_state)

    # 将返回的图状态进行json序列化
    json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logging.getLogger().info(json_state)
