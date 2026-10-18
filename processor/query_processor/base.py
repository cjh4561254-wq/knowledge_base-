# processor/query_processor/base.py
"""
检索流程节点基类。

约定（与 nodes/ 下 6 个检索节点、web/api/query_service.py 一致）：
  1. 子类覆盖 name，实现 process(state)；组图时 add_node(name, 实例) 走 __call__。
  2. 打点：进入节点先 add_running_task(session_id, name, is_stream)，
     各节点 process 内部自行 add_done_task；基类只负责 running 与异常兜底清理。
  3. 异常路径必须把本节点从 running 列表摘掉（对照导入侧
     processor/import_processor/base.py 的成品写法）：否则任何一个节点抛错，
     前端轮询 /stream 的进度条会永远卡在「XX执行中」，SSE 会话无法收口。
"""
from abc import ABC, abstractmethod
from typing import TypeVar

from tool.logger import logger
from utils.task_utils import add_done_task, add_running_task, set_task_result

# 定义泛型
T = TypeVar("T")


class NodeBase(ABC):

    # 节点名称：占位符
    # 子类中要覆盖这个名字
    name: str = "base_node"

    def __call__(self, state: T) -> T:
        # session_id 用 .get 而不是下标：节点单测（if __name__ == "__main__"）
        # 常常只传业务字段，缺 session_id 时直接 KeyError 会把「没打点」
        # 误报成「节点跑挂了」；缺 id 就跳过打点即可。
        session_id = state.get("session_id") if isinstance(state, dict) else None
        is_stream = state.get("is_stream") if isinstance(state, dict) else None
        tracked = bool(session_id)

        try:
            logger.info(f"{self.name} 开始执行")
            # 开始：记录节点运行状态（小本本上记录 当前session_id 哪个节点正在运行 是否流式）
            if tracked:
                add_running_task(session_id, self.name, is_stream)
            result = self.process(state)

            logger.info(f"{self.name} 结束执行")
            return result

        except Exception as e:
            logger.exception(f"{self.name} 执行异常: {e}")

            # 异常路径的打点闭环（关键修复）：
            # add_running_task 之后一旦抛错又原样上抛，running 列表里永远留着本节点，
            # 前端进度条卡死。用 add_done_task 把它从 running 移到 done（语义=已离开
            # 运行态），并 set_task_result 落一条 error 供 web 层取回展示失败原因。
            # 清理动作本身绝不能再把真正的业务异常盖掉，所以整体再包一层 try。
            if tracked:
                try:
                    set_task_result(session_id, "error", f"[{self.name}] {e}")
                    add_done_task(session_id, self.name, bool(is_stream))
                except Exception as clean_error:  # noqa: BLE001 —— 兜底清理，禁止影响主异常
                    logger.warning(f"{self.name} 失败打点清理异常：{clean_error}")
            raise

    # 将来 你的子类节点想要继承NodeBase 就必须实现 process方法
    @abstractmethod
    def process(self, state: T) -> T:
        pass
