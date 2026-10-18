# processor/import_processor/nodes/node_entry.py
"""
入口节点（任务分发）

职责：
  1. 从 state 取上传文件路径 import_file_path，做存在性/类型校验；
  2. 判定 PDF 还是 Markdown，写入路由开关 is_pdf_read_enabled / is_md_read_enabled
     以及对应路径 pdf_path / md_path，供 main_graph.route_after_entry 使用；
  3. 补齐任务上下文：task_id（打点用）、file_dir（PDF 转换与图片落盘目录）、
     file_title（下游 Milvus 幂等键）。

注意 Windows 路径：本项目全部按「原生 Windows 路径」约定
  DATA_BASED_ROOT_DIR=D:\file 、MD_ROOT_DIR=./temp-files/
Web 层传进来的就是 D:\\... 形式；但从 Git Bash 里手工跑节点单测时常见 /d/file/... 形式，
pathlib 在 Windows 下不会把它正确解析到 D 盘（会被当成「当前盘根的 /d/file」），
所以入口处统一归一化一次，避免整条流水线在错误路径上空转。
"""
import json
import logging
import os
import re
import uuid
from pathlib import Path

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import (
    FileProcessingError,
    StateFieldError,
    ValidationError,
)
from processor.import_processor.state import ImportGraphState

# 允许导入的文档类型 -> 对应的路由开关/路径字段（新增类型只改这张表）
_TYPE_ROUTES = {
    ".pdf": ("is_pdf_read_enabled", "pdf_path"),
    ".md": ("is_md_read_enabled", "md_path"),
}

# Git Bash / MSYS 风格路径：/d/xxx 或 /cygdrive/d/xxx
_MSYS_PATH_PATTERN = re.compile(r"^/(?:cygdrive/)?([a-zA-Z])(/.*)?$")

# 入口节点： 判断是PDF 还是 MD  决定下一个 去哪个节点工作

class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):
        # 1.从state中获取文件的绝对路径
        import_file_path = state.get("import_file_path")
        # 路径非空校验
        if not import_file_path:
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                message="文件路径不能为空",
                expected_type=str)

        #2.将import_file_path转换为Path对象（先归一化 Git Bash 风格路径）
        normalized_path = self._normalize_path(str(import_file_path))
        import_file_path_obj = Path(normalized_path)

        if not import_file_path_obj.is_file():# 文件存在校验（必须是文件，目录不算）
            raise FileProcessingError(
                node_name=self.name,
                message=f"文件{import_file_path_obj.name}不存在")

        # 3.判断文件类型（后缀大小写不敏感：.PDF/.MD 也要能进来）
        route = _TYPE_ROUTES.get(import_file_path_obj.suffix.lower())
        if route is None:
            raise ValidationError(
                node_name=self.name,
                message=f"不支持的文件类型{import_file_path_obj.suffix}，仅支持 .pdf / .md")

        # 两个开关都要显式写值：条件路由是先看 PDF 再看 MD，
        # 只写一个的话，重复执行/断点续跑时残留的另一个开关会把流程带偏。
        path_field = route[1]
        for flag, _field in _TYPE_ROUTES.values():
            state[flag] = (flag == route[0])
        state[path_field] = normalized_path

        #4. 获取文件名作为标题
        state["file_title"] = import_file_path_obj.stem

        # 5.补 task_id：Web 层一定会传；手工单节点跑（或主图 __main__）没传时生成一个，
        #   这样 BaseNode 的 running/done 打点依然生效，进度不会丢。
        if not state.get("task_id"):
            state["task_id"] = str(uuid.uuid4())

        # 6.补输出目录：PDF 转换节点强制要求 file_dir 非空，缺失时兜底到 MD_ROOT_DIR
        #   （.env 已由 config_base 的 load_dotenv 注入 os.environ），并直接把目录建好，
        #   免得下游节点因为目录不存在再报一次「看不懂的」FileProcessingError。
        file_dir = state.get("file_dir") or os.getenv("MD_ROOT_DIR", "./temp-files/")
        file_dir = self._normalize_path(str(file_dir))
        state["file_dir"] = file_dir
        try:
            Path(file_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # 建目录失败只降级告警：node_pdf_to_md 里还会再校验并给出用户可读错误
            self.logger.warning(f"输出目录{file_dir}创建失败，将由下游节点再次尝试：{e}")

        return state

    @staticmethod
    def _normalize_path(raw_path: str) -> str:
        """
        把路径统一成「原生 Windows 绝对/相对路径」字符串。

        为什么需要：Python 是原生 Windows 进程，而课程/调试常用 Git Bash，
        /d/file/a.pdf 这类 MSYS 路径交给 pathlib 会被当成当前盘根下的 \\d\\file，
        于是 exists()=False，报「文件不存在」。这里在入口一次性转掉。
        """
        path_text = (raw_path or "").strip().strip('"')
        if not path_text:
            return path_text

        # /cygdrive/d/xxx 与 /d/xxx → D:\xxx
        matched = _MSYS_PATH_PATTERN.match(path_text)
        if matched:
            drive, rest = matched.group(1).upper(), matched.group(2) or "\\"
            return f"{drive}:" + rest.replace("/", "\\")

        # 其余（D:\\x、\\\\服务器\\share、相对路径 ./temp-files/）保持原样，交给 pathlib 处理
        return path_text


if __name__ == "__main__":
    # 激活日志的全局配置
    setup_logging()

    # 初始化图状态
    init_state = {
        "import_file_path": r"D:\file\入口节点自测\hak180产品安全手册.pdf"
    }

    # 创建节点对象
    node_entry = NodeEntry()
    # 执行节点的单元测试
    result = node_entry(init_state)
    # 将返回的图状态进行json序列化
    json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logging.getLogger().info(json_state)
