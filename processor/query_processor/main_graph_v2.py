# processor/query_processor/main_graph_v2.py
"""
检索主图 V2（课程后期版本）。

与 V1（main_graph.py，web/api/query_service.py 实际使用的版本）的关系：
    V2 去掉了 node_multi_search / node_join 两个虚拟节点，直接让
    _route_after_item_name_confirm 返回「节点名列表」做并行 fan-out，
    三路检索节点用普通边汇聚到 node_rrf 完成隐式 fan-in。
    语义与 V1 拓扑等价，web 层仍调用 V1；本文件保持可 import、可组图，
    供对照学习与后续切换。

惰性导入说明同 V1：节点模块延后到 _init_nodes() 内 import，
避免只加载本文件就把 FlagEmbedding/torch、pymilvus 等重依赖拉进 sys.modules。
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


class KBQueryWorkflowV2:
    """
    知识库查询工作流类
    封装LangGraph工作流的构建、编译、执行逻辑，支持自定义配置和多实例运行
    """

    def __init__(self):
        """初始化工作流：创建状态图、注册节点、定义路由规则"""
        # 1. 初始化LangGraph状态图
        self.workflow = StateGraph(QueryGraphState)
        # 2. 初始化所有业务节点（实例属性，支持多实例隔离；节点类在此惰性 import）
        self._init_nodes()
        # 3. 注册节点到工作流
        self._register_nodes()
        # 4. 设置入口和路由规则
        self._setup_routes()
        # 5. 编译工作流（懒加载，首次执行时编译）
        self._compiled_app: object | None = None

    def _init_nodes(self):
        """初始化所有业务节点（私有方法，封装节点创建逻辑）"""
        # 惰性导入：见模块 docstring
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

        self.node_item_name_confirm = NodeItemNameConfirm()
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
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm) # 确认主体
        self.workflow.add_node("node_search_embedding", self.node_search_embedding) # 向量搜索
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde) #假设性答案向量搜索
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp) # 联网搜索
        self.workflow.add_node("node_rrf", self.node_rrf) # 排序
        self.workflow.add_node("node_rerank", self.node_rerank) # 重排
        self.workflow.add_node("node_answer_output", self.node_answer_output) # 生成

    def _route_after_item_name_confirm(self, state: QueryGraphState):
        """
        主体名称确认后的条件路由函数。
        V2 用「返回列表」直接 fan-out 三路并行检索（无需 node_multi_search 虚拟分叉点），
        langgraph 1.x 的 add_conditional_edges 原生支持路由函数返回 list[str]
        （已在 1.2.15 微图实测：同 superstep 并行执行，三路各写自己的 state key 不冲突）。
        """
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

        # 否则继续搜索流程（返回列表 = 三节点同 superstep 并行）
        return ["node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"]

    def _setup_routes(self):
        """设置工作流路由规则"""
        # 1、设置入口节点
        self.workflow.set_entry_point("node_item_name_confirm")

        # 2、注册条件路由边。
        # 路由函数返回「节点名列表」实现三路并行 fan-out（langgraph 1.x 原生支持，
        # 已在 1.2.15 上用微图实测：同 superstep 并行执行、各自写不同 state key 不冲突）。
        # path_map 字典必须保留：add_conditional_edges 靠它推断分支终点 ends，
        # 不传的话 get_graph() 会把并行分支误画成 n1 -> END，图可视化/排查全部失真。
        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "node_answer_output": "node_answer_output",
                "node_search_embedding": "node_search_embedding",
                "node_search_embedding_hyde": "node_search_embedding_hyde",
                "node_web_search_mcp": "node_web_search_mcp"
            }
        )

        # 3. 多路搜索结果合并：三条边汇入 node_rrf，LangGraph 自动等待全部完成（隐式 join）
        self.workflow.add_edge("node_search_embedding", "node_rrf")
        self.workflow.add_edge("node_search_embedding_hyde", "node_rrf")
        self.workflow.add_edge("node_web_search_mcp", "node_rrf")

        # 4. 排序 -> 重排 -> 生成 -> 结束
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
        :return: stream=True 返回 stream_mode="updates" 的生成器；stream=False 返回最终状态 dict。
        """
        if not self._compiled_app:
            self.compile()

        # 注意：这里刻意不做 print_ascii —— 旧版每次 run 都往控制台打印整张图，
        # 既污染服务日志，Windows GBK 控制台还可能在框线字符上 UnicodeEncodeError。
        # 需要看拓扑时用 __main__ 里的 draw_ascii() 或单独调用 get_graph()。

        if stream:
            # 显式 updates 模式，与导入侧及 main_graph(V1) 保持一致：逐节点产出 {节点名: 增量}
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

    workflow = KBQueryWorkflowV2()
    # 打印编译后的图结构（组图即可验证拓扑，不需要真实执行）
    logger.info(workflow.compile().get_graph().draw_ascii())
    # 流式输出
    # 注意：真跑需要 Milvus/DeepSeek 在线，联调时再打开
    # for chunk in workflow.run(init_state, stream=True):
    #     logger.warning(chunk)
