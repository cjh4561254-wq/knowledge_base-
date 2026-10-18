# tool/logger.py
"""
全局日志配置（导入 / 检索 / Web 服务共用同一个根 logger）。

为什么要做防御性配置（而不是课程骨架里的 4 行裸配置）：
1. Windows 控制台默认代码页是 GBK(cp936)，且服务常常在"无 TTY"环境运行
   （输出被重定向到文件 / uvicorn 后台）。中文与特殊字符可能触发
   UnicodeEncodeError —— logging 虽然会吞掉异常，但日志会直接丢失。
   因此强制把输出流编码设为 UTF-8，并用 backslashreplace 兜底，保证
   任何字符都不会因编码问题抛异常。
2. 无 TTY 时禁用 ANSI 彩色，避免重定向文件里出现 ^[[32m 之类的转义垃圾。
3. 模块可能被多次 import（uvicorn --reload / 测试脚本重复初始化），
   通过 handler 打标去重，防止日志重复输出（一次进程内只注册一次）。
4. 控制台乱码时排查困难，因此额外落一份 UTF-8 文件日志（logs/app.log），
   路径锚定项目根目录（基于 __file__ 推导），不依赖进程 cwd。
"""

import logging
import sys
from pathlib import Path

import colorlog

# 输出格式与课程骨架保持一致
_FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",  # INFO 显示为绿色
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

# 日志文件锚定项目根目录：基于 __file__ 推导，避免 cwd 不同导致日志文件漂移
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_FILE = _PROJECT_ROOT / "logs" / "app.log"

# 初始化全局日志对象（根 logger，全项目共享）
logger = logging.getLogger()
# 设置日志的默认的级别（骨架课程用 INFO；检索调试可按需 setup_logging(DEBUG)）
logger.setLevel(logging.INFO)


def _safe_reconfigure(stream):
    """
    把输出流统一为 UTF-8 + 容错编码。

    为什么：Windows GBK 控制台/重定向文件遇到非法字符会抛 UnicodeEncodeError，
    导致该条日志被 logging 悄悄丢弃；这里保证"永不因编码抛异常"。
    """
    if stream is None:
        return stream
    try:
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, OSError, ValueError):
        # pythonw 等无控制台环境下 stderr 可能不可重configure，静默降级
        pass
    return stream


def _supports_color(stream) -> bool:
    """仅当真终端才输出彩色；无 TTY（重定向/服务后台）用纯文本。"""
    try:
        return stream is not None and stream.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def _build_console_handler() -> logging.Handler:
    """加载彩色日志处理器（非 TTY 环境自动退化为纯文本，避免 ANSI 乱码）。"""
    handler = colorlog.StreamHandler(_safe_reconfigure(sys.stderr))
    if _supports_color(handler.stream):
        # 定义日志输出的格式（带颜色）
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s" + _FORMAT,
                datefmt=_DATEFMT,
                log_colors=_LOG_COLORS,
            )
        )
    else:
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    return handler


def _build_file_handler() -> logging.Handler | None:
    """文件日志：UTF-8 落盘，控制台乱码时可事后翻查；失败不影响启动。"""
    try:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_LOG_FILE, encoding="utf-8", delay=True)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        return handler
    except OSError:
        return None


def _add_handler_once(tag: str, handler: logging.Handler | None) -> None:
    """按 tag 去重注册：同一进程内重复 import / 重复 setup 不会重复输出。"""
    if handler is None:
        return
    for existing in logger.handlers:
        if getattr(existing, "_kb_tag", None) == tag:
            return
    handler._kb_tag = tag  # type: ignore[attr-defined]
    logger.addHandler(handler)


# 应用日志配置信息（幂等：重复执行不会注册重复 handler）
_add_handler_once("console", _build_console_handler())
_add_handler_once("file", _build_file_handler())


def setup_logging(level: int = logging.INFO) -> None:
    """
    调整日志级别（供调试入口显式调用，例如 setup_logging(logging.DEBUG)）。

    为什么提供函数而不是在模块顶层设 DEBUG：库被导入时不应该自作主张地
    把全局级别改高，噪音日志会影响 FastAPI / LangGraph 服务的可读性。
    """
    logger.setLevel(level)
