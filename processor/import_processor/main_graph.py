# processor/import_processor/main_graph.py
"""
导入功能的主流程图（LangGraph 编排层）

职责：只做「编排」——把 7 个业务节点连成课程定义的导入流水线，
负责入口路由、测试旁路、图级别日志/耗时统计与异常冒泡，
不含任何业务逻辑（业务逻辑都在各自的 BaseNode 子类里）。

图顺序（课程 day13 定义，务必与下面常量一致）：
    START
      -> node_entry                  入口：判断文档类型
      -> node_pdf_to_md              PDF 转 Markdown
      -> node_md_img                 MD 图片处理（视觉模型 + 对象存储）
      -> node_document_split         文档切片
      -> node_item_name_recognition  主体（商品名）识别
      -> node_bge_embedding          向量化
      -> node_import_milvus          写入 Milvus
      -> END
入口节点之后走条件边：PDF -> node_pdf_to_md；MD -> node_md_img（跳过 PDF 转换）；
其他格式 -> END（直接结束，不入库）。

版本说明（本机装的是 langgraph 1.2.15 / langchain 1.4.6，即 1.x）：
    * 0.x 教程里的 `from langgraph.constants import END` 仍可用，但 1.x 的正式导出位置
      是 `langgraph.graph`，本文件统一用 `from langgraph.graph import START, END`。
    * `set_entry_point("xxx")` 在 1.x 仍保留（内部等价于 add_edge(START, "xxx")），
      这里用更显式的 `add_edge(START, NODE_ENTRY)`，避免依赖已淡出的老 API。
    * `compile()` 的参数只剩 checkpointer/cache/store/interrupt_* 等，
      老的 `compile(debug=True)` 之类写法没影响；`invoke/stream` 签名与 0.x 一致。
"""

import logging
import os
import sys
import time
import uuid

from dotenv import load_dotenv

# 为什么必须 override=True：Windows 系统环境变量里残留了一个**已失效**的
# moonshot OPENAI_API_KEY / OPENAI_BASE_URL，进程环境优先级高于 .env，
# 不覆盖的话所有大模型调用都会 401（这是本项目的既定修复方案，别改回 load_dotenv()）。
# 也正因为要先落地环境变量，后面的 import 才写在下面（ruff/pylint 的 E402 为此放行）。
load_dotenv(override=True)

from langgraph.graph import END, START, StateGraph

from processor.import_processor.base import setup_logging
from processor.import_processor.exceptions import ImportProcessError
from processor.import_processor.state import ImportGraphState, get_default_state
from utils.task_utils import set_task_result

logger = logging.getLogger("import.main_graph")

# ==================== 图结构契约（顺序 = 课程定义） ====================
NODE_ENTRY = "node_entry"
NODE_PDF_TO_MD = "node_pdf_to_md"
NODE_MD_IMG = "node_md_img"
NODE_DOCUMENT_SPLIT = "node_document_split"
NODE_ITEM_NAME_RECOGNITION = "node_item_name_recognition"
NODE_BGE_EMBEDDING = "node_bge_embedding"
NODE_IMPORT_MILVUS = "node_import_milvus"
NODE_TEST = "node_test"

# 真实链路的节点执行顺序（__main__ 与自检脚本用它做结构校验）
EXPECTED_PIPELINE: tuple[str, ...] = (
    NODE_ENTRY,
    NODE_PDF_TO_MD,
    NODE_MD_IMG,
    NODE_DOCUMENT_SPLIT,
    NODE_ITEM_NAME_RECOGNITION,
    NODE_BGE_EMBEDDING,
    NODE_IMPORT_MILVUS,
)

# 「测试节点」旁路开关的环境变量名：取值 1/true/yes/on 时，整张图退化为
# START -> node_test -> END，用于本地单步调试编排逻辑（不碰模型/MinIO/Milvus）。
# 默认（未设置）走真实链路，线上行为不受影响。
TEST_MODE_ENV = "KB_IMPORT_TEST_MODE"


def _env_truthy(name: str, default: bool = False) -> bool:
    """读取布尔型环境变量（避免 "False"/"0" 被 bool(str) 误判为真）。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


class KBImportWorkflow:
    """
    导入工作流：LangGraph 图的构建 + 编译 + 执行入口。

    对外契约（web/api/import_service.py 依赖，勿改）：
        workflow = KBImportWorkflow()
        workflow.run(init_state)                 # 阻塞执行，返回最终 state
        for event in workflow.run(state, True):  # 流式，逐节点产出 {节点名: state 增量}
    """

    def __init__(self, test_mode: bool | None = None):
        """
        初始化工作流对象。

        :param test_mode: True/False 强制开/关「测试节点旁路」；
                          None（默认）= 读环境变量 KB_IMPORT_TEST_MODE，即默认走真实链路。
        """
        self._compiled_graph = None
        self.test_mode = _env_truthy(TEST_MODE_ENV) if test_mode is None else bool(test_mode)

    @property
    def graph(self):
        """
        懒加载：第一次真正用到时才实例化节点并编译图。
        为什么懒：编译期会 import 全部节点模块（pymilvus / langchain_openai 等），
        放到模块顶层会让 Web 服务启动就吞下这些重依赖。
        """
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    # ==================== 组图 ====================

    def build_graph(self):
        """
        构建并编译导入工作流。

        节点注册方式（重要约定）：把 BaseNode 子类的**实例**直接交给 add_node，
        LangGraph 调度时调用的是实例的 `__call__`（由基类统一做
        add_running_task/add_done_task 打点 + 异常包装成 ImportProcessError），
        再由 `__call__` 调子类的 `process`。
        所以这里**绝不能挂 node.process**——那样会绕过基类的任务追踪和结构化异常。
        """
        graph = StateGraph(ImportGraphState)

        if self.test_mode:
            # 旁路：只挂测试节点，用于验证编排/日志/状态传递，不执行真实业务
            from processor.import_processor.nodes.node_test import TestNode

            logger.warning(f"[测试旁路] 导入工作流仅执行 {NODE_TEST} 节点（{TEST_MODE_ENV}）")
            graph.add_node(NODE_TEST, self._instrument(TestNode()))
            graph.add_edge(START, NODE_TEST)
            graph.add_edge(NODE_TEST, END)
            return graph.compile()

        # 1. 局部导入：只有真正组图时才加载节点模块（见 graph 属性的「为什么懒」注释）
        from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
        from processor.import_processor.nodes.node_document_split import (
            NodeDocumentSplit,
        )
        from processor.import_processor.nodes.node_entry import NodeEntry
        from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
        from processor.import_processor.nodes.node_item_name_recognition import (
            NodeItemNameRecognition,
        )
        from processor.import_processor.nodes.node_md_img import NodeMDImg
        from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD

        # 2. 顺序 = 链路顺序，注册即文档（顺序本身由边决定，这里只为可读性）
        graph.add_node(NODE_ENTRY, self._instrument(NodeEntry()))
        graph.add_node(NODE_PDF_TO_MD, self._instrument(NodePDFToMD()))
        graph.add_node(NODE_MD_IMG, self._instrument(NodeMDImg()))
        graph.add_node(NODE_DOCUMENT_SPLIT, self._instrument(NodeDocumentSplit()))
        graph.add_node(NODE_ITEM_NAME_RECOGNITION, self._instrument(NodeItemNameRecognition()))
        graph.add_node(NODE_BGE_EMBEDDING, self._instrument(NodeBGEEmbedding()))
        graph.add_node(NODE_IMPORT_MILVUS, self._instrument(NodeImportMilvus()))

        # 3. 入口边：1.x 用 START 常量（set_entry_point 的老写法等价但已不推荐）
        graph.add_edge(START, NODE_ENTRY)

        # 4. 条件边：入口节点判断文档类型后决定走向
        graph.add_conditional_edges(
            NODE_ENTRY,
            self.route_after_entry,
            {
                # key = 路由函数返回值，value = 目标节点名
                NODE_PDF_TO_MD: NODE_PDF_TO_MD,
                NODE_MD_IMG: NODE_MD_IMG,
                END: END,
            },
        )

        # 5. 顺序边（课程定义的流水线）
        graph.add_edge(NODE_PDF_TO_MD, NODE_MD_IMG)
        graph.add_edge(NODE_MD_IMG, NODE_DOCUMENT_SPLIT)
        graph.add_edge(NODE_DOCUMENT_SPLIT, NODE_ITEM_NAME_RECOGNITION)
        graph.add_edge(NODE_ITEM_NAME_RECOGNITION, NODE_BGE_EMBEDDING)
        graph.add_edge(NODE_BGE_EMBEDDING, NODE_IMPORT_MILVUS)
        graph.add_edge(NODE_IMPORT_MILVUS, END)

        # 6. 编译（不传 checkpointer：导入流程是一次性批处理，无需断点续跑）
        return graph.compile()

    @staticmethod
    def route_after_entry(state: ImportGraphState) -> str:
        """
        决定入口节点之后的走向。

        :param state: 图状态（由 node_entry 写入 is_pdf_read_enabled / is_md_read_enabled）
        :return: 节点名，或 END 直接结束
        """
        # 与上游 state.py 的字段名严格对齐；缺字段按 False 处理（防御式，不让路由崩掉）
        if state.get("is_pdf_read_enabled"):
            return NODE_PDF_TO_MD
        if state.get("is_md_read_enabled"):
            return NODE_MD_IMG
        logger.warning("入口节点未识别到 PDF/MD 标记，流程直接结束（不支持的文件类型）")
        return END

    # ==================== 可观测性 ====================

    def _instrument(self, node):
        """
        给节点套一层「计时 + 起止日志」壳子，返回值仍是节点对 state 的处理结果。

        为什么不改基类：任务打点（add_running_task/add_done_task）是 BaseNode 的职责，
        图这一层只补充「哪个节点跑了多久、整条链跑到第几步」这类编排级信息，
        出异常时**原样抛出**——吞异常会让前端进度条永远停在 running。
        """

        def _node_with_timing(state: ImportGraphState):
            node_name = getattr(node, "name", "") or type(node).__name__
            task_id = state.get("task_id", "") if isinstance(state, dict) else ""
            started = time.perf_counter()
            logger.info(f"[开始] {node_name} task_id={task_id}")
            try:
                # 正常约定走 BaseNode.__call__（内部做打点 + 异常包装 + 调 process）；
                # 万一未来基类改成显式方法名，这里退回 run/process，保证图不会挂。
                run_node = node if callable(node) else (getattr(node, "run", None) or node.process)
                result = run_node(state)
            except Exception as exc:
                cost = time.perf_counter() - started
                # 基类已把底层异常包成 ImportProcessError；这里只补耗时维度
                logger.error(f"[失败] {node_name} 耗时={cost:.2f}s 原因={exc}")
                raise
            cost = time.perf_counter() - started
            logger.info(f"[完成] {node_name} 耗时={cost:.2f}s")
            return result

        # 保留可读的函数名，便于日志/调试时定位（LangGraph 用显式节点名，不依赖它）
        _node_with_timing.__name__ = f"{getattr(node, 'name', 'node')}_instrumented"
        return _node_with_timing

    @staticmethod
    def _normalize_state(state: ImportGraphState | dict | None) -> ImportGraphState:
        """
        补全初始状态：以 get_default_state() 为底，再用调用方传入的字段覆盖。

        为什么必要：BaseNode.__call__ 里直接读 `state["task_id"]` 做任务打点，
        手动入口/单测常常只传 import_file_path，缺 task_id 会先炸在打点上，
        报错信息还和真实问题无关。task_id 缺失时补一个 UUID。
        """
        init_state: ImportGraphState = get_default_state()
        if state:
            init_state.update(dict(state))  # type: ignore[arg-type]
        if not init_state.get("task_id"):
            init_state["task_id"] = str(uuid.uuid4())
        return init_state

    # ==================== 执行 ====================

    def run(self, state: ImportGraphState, stream: bool = False):
        """
        对外执行入口（导入 API 的后台任务、以及 __main__ 手动调试都走这里）。

        :param state: 初始状态；允许只传 import_file_path / file_dir / task_id 等必要字段
        :param stream: True=逐节点生成器；False=阻塞执行并返回最终 state
        :return: 最终状态 dict，或 stream_mode="updates" 的生成器
        :raises ImportProcessError: 任一节点失败时结构化冒泡（含 node_name / cause）
        """
        init_state = self._normalize_state(state)
        task_id = init_state["task_id"]

        if self.test_mode:
            logger.warning(f"[{task_id}] 使用【测试旁路】图：仅执行 {NODE_TEST} 节点")
        logger.info(
            f"[{task_id}] 导入工作流启动 模式={'stream' if stream else 'invoke'} "
            f"文件={init_state.get('import_file_path') or init_state.get('pdf_path') or init_state.get('md_path')}"
        )

        if stream:
            return self._run_stream(init_state, task_id)
        return self._run_invoke(init_state, task_id)

    def _run_invoke(self, init_state: ImportGraphState, task_id: str) -> ImportGraphState:
        """阻塞执行：整图跑完返回最终状态。"""
        started = time.perf_counter()
        try:
            final_state = self.graph.invoke(init_state)
        except ImportProcessError as exc:
            self._on_failure(task_id, exc, started)
            raise
        except Exception as exc:  # 非节点异常（路由/编译/状态问题）也要结构化冒泡，不能裸抛
            wrapped = ImportProcessError(
                message=f"导入工作流执行异常：{exc}", node_name="import_graph", cause=exc
            )
            self._on_failure(task_id, wrapped, started)
            raise wrapped from exc

        cost = time.perf_counter() - started
        logger.info(
            f"[{task_id}] 导入工作流全部完成 总耗时={cost:.2f}s "
            f"主体={final_state.get('item_name')!r} 切片数={len(final_state.get('chunks') or [])}"
        )
        return final_state

    def _run_stream(self, init_state: ImportGraphState, task_id: str):
        """
        流式执行：逐个产出「{节点名: 该节点对 state 的增量}」。

        为什么显式写 stream_mode="updates"：import_service.py 里是
        `for node_name, node_result in event.items()`，靠 update 事件拿到**节点名**
        去刷新前端进度；用 values 模式会退化成「整份 state 没有节点名」，前端进度条直接失效。
        """
        started = time.perf_counter()
        completed = 0
        try:
            for event in self.graph.stream(init_state, stream_mode="updates"):
                # event: {节点名: 状态增量}；END 之类的控制事件可能为空 dict
                for node_name in event:
                    completed += 1
                    logger.info(f"[{task_id}] 节点完成({completed}) {node_name}")
                yield event
        except ImportProcessError as exc:
            self._on_failure(task_id, exc, started)
            raise
        except Exception as exc:  # 同上：生成器里的异常也要结构化后再冒泡
            wrapped = ImportProcessError(
                message=f"导入工作流执行异常：{exc}", node_name="import_graph", cause=exc
            )
            self._on_failure(task_id, wrapped, started)
            raise wrapped from exc

        logger.info(f"[{task_id}] 导入工作流全部完成(流式) 总耗时={time.perf_counter() - started:.2f}s")

    def _on_failure(self, task_id: str, exc: ImportProcessError, started: float) -> None:
        """失败收口：落一条带节点/任务的结构化日志 + 记录错误结果，然后原样上抛（绝不吞）。"""
        logger.error(
            f"[{task_id}] 导入工作流失败 节点={exc.node_name} 耗时={time.perf_counter() - started:.2f}s "
            f"原因={exc.cause or '-'} 信息={exc}"
        )
        # 写进任务结果表，便于 /status 或后续接口回查具体失败原因（不影响调用方既有逻辑）
        set_task_result(task_id, "error", str(exc))

    # ==================== 结构自检辅助 ====================

    def node_names(self) -> list[str]:
        """编译后图里的业务节点名（去掉 START/END 控制节点），用于顺序校验/日志。"""
        return [name for name in self.graph.get_graph().nodes if name not in (START, END)]

    def describe(self) -> str:
        """人类可读的图结构（Mermaid 兜底 ASCII），排查连线问题时先看这个。"""
        drawable = self.graph.get_graph()
        try:
            return drawable.draw_ascii()
        except Exception:  # noqa: BLE001  # draw_ascii 依赖 grandalf，缺失时降级为 mermaid，不让自检崩掉
            return drawable.draw_mermaid()


# ==================== 手动入口：跑一条真实 PDF（调试用） ====================
if __name__ == "__main__":
    # 激活全局日志（colorlog，INFO 级别）
    setup_logging()

    # 用法：
    #   python -m processor.import_processor.main_graph <pdf/md 路径> [输出目录] [--test] [--stream]
    #   KB_IMPORT_TEST_MODE=1 python -m processor.import_processor.main_graph   # 只验证编排
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a.lstrip("-") for a in sys.argv[1:] if a.startswith("--")}
    use_stream = "stream" in flags

    debug_file_path = args[0] if args else os.getenv("KB_IMPORT_DEBUG_FILE", "")
    debug_file_dir = args[1] if len(args) > 1 else (os.getenv("MD_ROOT_DIR", "./temp-files/") or "./temp-files/")

    # 只在显式传 --test 时强制开旁路；不传则交由环境变量 KB_IMPORT_TEST_MODE 决定
    workflow = KBImportWorkflow(test_mode=True if "test" in flags else None)

    # 先把图结构打出来：连线错没错，一眼能看出来，不用等跑完
    logger.info(f"节点顺序：{workflow.node_names()}")
    logger.info(f"图结构：\n{workflow.describe()}")

    if workflow.test_mode:
        # 旁路模式不需要任何文件，直接空跑图（task_id 由 _normalize_state 补）
        result_state = workflow.run({"task_id": "main_graph_test"}, stream=use_stream)
        logger.info(f"测试结果：{result_state if not use_stream else list(result_state)}")
    else:
        if not debug_file_path:
            raise SystemExit("请传入要导入的文件路径：python -m processor.import_processor.main_graph <pdf路径>")
        result_state = workflow.run(
            {
                "task_id": str(uuid.uuid4()),
                "import_file_path": debug_file_path,
                "file_dir": debug_file_dir,
            },
            stream=use_stream,
        )
        logger.info(f"最终状态：{result_state if not use_stream else list(result_state)}")
