# processor\import_processor\nodes\node_test.py
"""
骨架课程（day13/03）自带的示例节点：用来验证 BaseNode 的日志/打点/异常包装是否工作。

它没有注册进 main_graph.py，也不读写任何业务 state 字段，
所以「能 import、能单独跑、不干扰主图」就是它的唯一职责。
"""
import json
import logging
from typing import Any, Dict

from processor.import_processor.base import BaseNode, setup_logging


class TestNode(BaseNode):

    name = "node_test"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """演示基类能力：分级日志 + log_step；返回值原样带回，不改任何业务字段。"""

        self.logger.error(f"{self.name}正在执行")
        self.logger.debug(f"{self.name}正在debug")
        self.logger.warning(f"{self.name}warning")

        self.log_step("MCP查询", "开始执行")

        return state


if __name__ == "__main__":

    # 激活日志
    setup_logging(logging.INFO)

    test_node = TestNode()
    # 使用 对象() 的方式相当于调用了 对象的__call__()
    # 传 task_id 才能看到 running/done 打点的完整闭环
    result = test_node({"abc": 456, "task_id": "node_test_demo"})
    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
