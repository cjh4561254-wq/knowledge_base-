# 主图
# processor/query_processor/main_graph.py
"""
检索主图（课程 V1 版，web/api/query_service.py 实际调用的就是本文件的 KBQueryWorkflow）。

拓扑（对照课程《11 骨架代码》《12 产品确认》《16 结果融合重排》）：
    START -> node_item_name_confirm
              |- 有 answer（追问/拒答） -> node_answer_output -> END
              |- 无 answer -> node_multi_search（虚拟分叉点）
                               ├-> node_search_embedding      ┐
                               ├-> node_search_embedding_hyde ├-> node_join（虚拟合并点）
                               └-> node_web_search_mcp        ┘
           node_join -> node_rrf -> node_rerank -> node_answer_output -> END

「是否需要追问确认商品名」不走 interrupt：node_item_name_confirm 把反问/拒答话术
写进 state["answer"]，由 _route_after_item_name_confirm 条件路由直接跳 answer 输出。

惰性导入说明（为什么节点模块不在文件顶层 import）：
    检索节点会连带 import utils.embedding_utils / utils.milvus_utils 等，
    重型依赖（FlagEmbedding/torch、pymilvus）虽有惰性 getter，但顶层 import
    仍会让只加载本模块的调用方（如 web 层启动、单元测试）白白吃内存并拖慢启动。
    因此节点类统一延后到 _init_nodes() 内 import + 实例化。
"""
from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.query_processor.state import QueryGraphState
from tool.logger import logger

# 初始化环境变量（类加载前执行，保证全局生效）
# 必须 override=True：系统环境变量里残留一个已失效的 OPENAI_API_KEY/BASE_URL，
# 不覆盖会把 DeepSeek 配置污染掉。
load_dotenv(override=True)


class KBQueryWorkflow:
    """
    知识库查询工作流类
    封装LangGraph工作流的构建、编译、执行逻辑，支持自定义配置和多实例运行
    """

    def __init__(self):
        """初始化工作流：创建状态图、注册节点、定义路由规则"""
        # 1. 初始化LangGraph状态图
        self.workflow = StateGraph(QueryGraphState) # 创建图
        # 2. 初始化所有业务节点（实例属性，支持多实例隔离；节点类在此惰性 import）
        self._init_nodes()
        # 3. 注册节点到工作流
        self._register_nodes()
        # 4. 设置入口和路由规则
        self._setup_routes()
        # 5. 编译工作流（懒加载，首次执行时编译）
        self._compiled_app = None

    def _init_nodes(self):
        """初始化所有业务节点（私有方法，封装节点创建逻辑）"""
        # 惰性导入：见模块 docstring，避免 import 本文件就把检索链路的重依赖拉进来
        from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput
        from processor.query_processor.nodes.node_item_name_confirm import (
            NodeItemNameConfirm,
        )
        from processor.query_processor.nodes.node_rerank import NodeRerank
        from processor.query_processor.nodes.node_rrf import NodeRrf
        from processor.query_processor.nodes.node_search_embedding import (
            NodeSearchEmbedding,
        )
        from processor.query_processor.nodes.node_search_embedding_hyde import (
            NodeSearchEmbeddingHyde,
        )
        from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp

        self.node_item_name_confirm = NodeItemNameConfirm()# 实例化
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    def _register_nodes(self):
        """注册所有节点到工作流（私有方法，统一管理节点注册）"""
        # 节点标识与实例属性名保持一致，便于维护；
        # 挂的是 NodeBase 子类实例，LangGraph 调用时走基类 __call__（统一打点/异常清理）。
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm)  # 确认主体
        self.workflow.add_node("node_multi_search", lambda x: x)  # 虚拟节点：多路搜索分叉点（状态不变）
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)  # 向量搜索
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)  # 假设性答案向量搜索
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp)  # 联网搜索
        self.workflow.add_node("node_join", lambda x: {})  # 虚拟节点：多路搜索合并点
        self.workflow.add_node("node_rrf", self.node_rrf)  # 排序
        self.workflow.add_node("node_rerank", self.node_rerank)  # 重排
        self.workflow.add_node("node_answer_output", self.node_answer_output)  # 生成

        # 虚拟节点的作用：作为流程的「分叉 / 合并中转站」，解决多分支流程的组织问题，本身无业务逻辑；
        # lambda x:x 含义：接收 state 并原样返回，是最轻便的 “无逻辑传递” 方式；
        # 普通函数替换：定义 def 函数名(state): return state 即可完全等价，优势是易扩展、易调试；
        # 注意：并行三节点各自只返回自己字段的增量 dict（如 {"embedding_chunks": [...]}），
        # 同一 superstep 里写同一个 LastValue key 会报 InvalidUpdateError，所以不许改成返回全量 state。

    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str:
        """主体名称确认后的条件路由函数"""
        if state.get("answer"):
            """
            这主要发生在 node_item_name_confirm 节点无法直接确定唯一的商品型号，从而需要“反问用户”或“拒绝回答”的场景。
            具体来说，有以下两种情况会导致 state 中直接出现 answer ，从而跳过后续的检索流程，直接输出：
            1. 多选一（反问用户） ：
            - 场景 ：用户问得太模糊（比如“华为P60”），系统发现数据库里有“华为P60 128G”和“华为P60 Art”两个型号，不足以直接确认。
            - 处理 ：节点会生成一条反问句作为 answer ，例如：“您是想问以下哪个产品：华为P60 128G、华为P60 Art？请明确一下型号。”
            - 结果 ：此时不需要去检索文档了，直接把这句话发给用户让他选。

            2. 查无此人（拒绝回答） ：
            - 场景 ：用户问了一个系统里压根没有的商品（比如“小米15”，库里只有华为的数据）。
            - 处理 ：节点会生成一条拒绝句作为 answer ，例如：“抱歉，未找到相关产品，请提供准确型号以便我为您查询。”
            - 结果 ：同样不需要后续检索，直接结束流程。
            """
            return "node_answer_output"

        # 否则继续搜索流程
        return "node_multi_search"

    def _setup_routes(self):
        """设置工作流路由规则"""
        # 1、设置入口节点
        self.workflow.set_entry_point("node_item_name_confirm")

        # 2、注册条件路由边
        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,# 完善 决定 意图识别节点 后面的走向
            {
                "node_answer_output": "node_answer_output",
                "node_multi_search": "node_multi_search"
            }
        )

        # 3. 并发执行搜索
        self.workflow.add_edge("node_multi_search", "node_search_embedding")
        self.workflow.add_edge("node_multi_search", "node_search_embedding_hyde")
        self.workflow.add_edge("node_multi_search", "node_web_search_mcp")

        # 4. 多路搜索结果合并
        self.workflow.add_edge("node_search_embedding", "node_join")
        self.workflow.add_edge("node_search_embedding_hyde", "node_join")
        self.workflow.add_edge("node_web_search_mcp", "node_join")

        # 5. 合并 -> 排序 -> 重排 -> 生成 -> 结束
        self.workflow.add_edge("node_join", "node_rrf")
        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)

    def compile(self):
        """编译工作流（公开方法，支持手动触发编译）"""
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: QueryGraphState, stream: bool = False):
        """
        统一执行入口，支持切换invoke/stream
        :param initial_state:  初始状态对象
        :param stream: 是否是流式输出
        :return: stream=True 返回 stream_mode="updates" 的生成器（逐节点 {节点名: 增量}）；
                 stream=False 返回执行完成后的状态 dict。
        """
        if not self._compiled_app:
            self.compile()
        if stream:
            # 显式 updates 模式：与导入侧 KBImportWorkflow._run_stream 对齐。
            # SSE 进度帧虽然主要由 task_utils 直推队列，但 updates 事件带节点名，
            # 便于日志/后续按节点消费；默认 values 模式只有整份 state、没有节点名。
            return self._compiled_app.stream(initial_state, stream_mode="updates")
        else:
            return self._compiled_app.invoke(initial_state)


# ===================== 用法示例 =====================

if __name__ == "__main__":

    # 定义初始状态
    init_state = {
        "session_id": "local_debug",
        "original_query": "如何使用万用表测量电压？",
        "is_stream": False,
    }

    workflow = KBQueryWorkflow()
    # 打印编译后的图结构（组图即可验证拓扑，不需要真实执行）
    logger.info(workflow.compile().get_graph().draw_ascii())
    # 流式输出
    # run 给你调用  run 执行 我给你的工作流
    # 注意：真跑需要 Milvus/DeepSeek 在线，联调时再打开
    # for chunk in workflow.run(init_state, stream=True):
    #     logger.warning(chunk)
