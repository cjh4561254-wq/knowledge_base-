# knowledge/processor/import_processor/base.py

"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能：
  1. 统一日志（节点开始/结束、步骤日志）
  2. 统一任务打点（running / done，供前端进度条轮询 /status/{task_id}）
  3. 统一异常包装（任何失败都转成 ImportProcessError 子类，携带节点名与原始异常）
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, TypeVar

import colorlog

from processor.import_processor.config_base import ImportConfig, get_config
from processor.import_processor.exceptions import ImportProcessError
from utils.task_utils import add_done_task, add_running_task, set_task_result

"""
1、抽象的类（ABC），子类需要实现这个类
2、抽象的方法（@abstractmethod process），子类需要实现这个方法
3、name属性，子类必须覆盖这个属性
4、__init__() 方法中初始化了日志记录器，日志记录器的名字是 self所在类的名字
5、__init__() 方法中初始化了全局配置，获取全局单例配置对象
6、__call__() 方法的调用时机  
    (1)对象 = 类名() 
    (2)结果 = 对象()     圆括号调用了__call__()方法
7、self.process()， 当前这句话是从哪调用过来的，process()就定义在哪个位置
8、未来可以在父类的process()中实现任务追踪
9、封装了统一日志处理，使用logging、colorlog
    注意：使用日志记录器记录日志时一定要先激活日志setup_logging(logging.INFO)，并指定日志级别

"""
T = TypeVar("T")  # 泛型状态类型

class BaseNode(ABC):
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        state = node()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config or get_config()

        # 日志记录器：命名 import.{self.name}
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """
        # task_id 用 .get 而不是 state["task_id"]：
        # 每个节点的 `if __name__ == "__main__"` 单测大多只传业务字段（md_path/chunks…），
        # 直接下标取会 KeyError，把「节点还没跑」误报成「节点跑挂了」，也让人无法单节点调试。
        # 缺 task_id 时跳过打点即可（打点只为 Web 进度条服务，与业务逻辑无关）。
        task_id = state.get("task_id") if isinstance(state, dict) else None
        # 打点开关：空字符串/None 都不打，避免往全局任务表里塞一个 "" 脏 key
        tracked = bool(task_id)

        try:
            # 1. 开始准备执行节点
            self.logger.info(f"--- {self.name} 开始啦 ---")

            # 开始：记录节点运行状态
            if tracked:
                add_running_task(task_id, self.name)

            # 2. 执行节点
            result = self.process(state)

            # 结束：记录节点完成状态
            if tracked:
                add_done_task(task_id, self.name)

            # 3. 执行节点成功
            self.logger.info(f"--- {self.name} 完成啦 ---")

            return result

        except Exception as e:
            # exception()：失败必须留完整堆栈，否则线上只剩一句 str(e) 无从定位
            self.logger.exception(f"{self.name} 执行失败: {e}")

            # 4. 异常路径的打点闭环（关键修复）：
            #    add_running_task 之后一旦抛错，之前的版本直接往上抛，running 列表里
            #    永远留着本节点 —— 前端轮询 /status/{task_id} 时进度条就卡死在这个节点。
            #    task_utils 没有「只摘 running 不进 done」的接口，这里用 add_done_task
            #    把节点从 running 移到 done（语义 = 已离开运行态），
            #    同时用 set_task_result 落一条 error，供 web 层取回展示失败原因。
            #    清理动作本身绝不能再抛异常盖掉真正的业务异常，所以整体再包一层 try。
            if tracked:
                try:
                    set_task_result(task_id, "error", f"[{self.name}] {e}")
                    add_done_task(task_id, self.name)
                except Exception as clean_error:  # noqa: BLE001 —— 兜底清理，禁止影响主异常
                    self.logger.warning(f"{self.name} 失败打点清理异常：{clean_error}")

            # 子类已经是 ImportProcessError（StateFieldError/MilvusError…）时原样抛出：
            # 再包一层会丢掉具体类型，web/main_graph 就没法按类型分级捕获用户可读错误了。
            # 只补一件事：节点自己 raise 时通常没传 node_name，这里回填，
            # 让前端拿到的文案带上「是哪个节点挂的」。
            if isinstance(e, ImportProcessError):
                if not getattr(e, "node_name", ""):
                    e.node_name = self.name
                raise

            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            ) from e

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑

        子类必须实现此方法。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志
            [步骤名称]传递进来的消息
        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


# 配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置日志格式

    Args:
        level: 日志级别
    """

    logger = logging.getLogger()
    logger.setLevel(level)

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',  # INFO 显示为绿色
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    ))

    logger.handlers.clear()
    logger.addHandler(handler)
