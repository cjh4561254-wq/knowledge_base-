# processor/query_processor/nodes/node_web_search_mcp.py
"""网络搜索节点（三路召回的第 3 路）。

本机链路变更说明（为什么默认不再是 MCP）：
- 课程版走阿里云百炼 DashScope MCP（bailian_web_search），但该 MCP key 本机实测 401，
  因此按契约改为 **Tavily**（tavily-python 0.8.5，已实测可用），
  引擎由 `config.bailian_mcp_config.mcp_config.web_search_engine` 开关切换
  （tavily 默认 / mcp 保留原分支代码），`tavily_api_key` 提供密钥。
- MCP 分支的 `from agents.mcp import ...` 改为**惰性 import**：默认走 Tavily 时不加载
  openai-agents，也避免该包缺失/改版时直接把这个节点打成 import 失败。

对外结构对齐（供 node_rrf / node_rerank 消费）：每条文档同时给出
「旧契约字段 title/url/snippet」（node_rerank._step_1_merge_multi_source_docs 在读）
和「三路统一字段 content/chunk_id/source/score/item_name + entity 子字典」
（node_rrf 对两路向量召回是取 doc["entity"]，网络路给同样的形状即可并入）。
chunk_id 用 url 哈希生成 `web_xxxx` 形式的稳定唯一键：本地两路的 RRF 去重键是 chunk_id，
若网络结果留 None 会被算成同一条而被相互覆盖。
"""

import hashlib
import json

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.task_utils import add_done_task

# 网络搜索返回条数（课程规格 count=5；Tavily 对应 max_results）
WEB_SEARCH_TOP_K = 5
# 单次网络请求超时（秒）：三路并行，网络路不能拖住整张检索图
WEB_SEARCH_TIMEOUT = 20
# 引擎开关取值
ENGINE_TAVILY = "tavily"
ENGINE_MCP = "mcp"

# Tavily 客户端单例：内部维护 requests.Session（连接复用），检索链路并发时只建一次
_tavily_client = None


def _get_tavily_client():
    """
    惰性获取 TavilyClient（import 本模块不要求已装 tavily，也不提前建连接）。
    :return: TavilyClient 实例；未配置 API Key 时返回 None（调用方降级为空结果）
    """
    global _tavily_client

    api_key = (mcp_config.tavily_api_key or "").strip()
    if not api_key:
        logger.error("未配置 TAVILY_API_KEY，网络搜索降级为空结果")
        return None

    if _tavily_client is None:
        from tavily import TavilyClient  # 延迟 import：仅在真正启用 tavily 时才依赖该包

        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def _build_web_doc(title: str, url: str, snippet: str, score=None) -> dict:
    """
    把一条网络结果规整成与本地两路召回同构的文档 dict。

    :param title: 结果标题
    :param url: 结果链接
    :param snippet: 摘要/正文片段（非空才有意义，调用方已过滤）
    :param score: 搜索引擎给出的相关度（Tavily 返回 0~1；MCP 分支无此字段则为 None）
    :return: 同时兼容 node_rerank（title/url/snippet）与 node_rrf（entity.chunk_id）的 dict
    """
    # 唯一去重键：同一 URL 多次出现只算一条，且与本地 chunk_id（int）不会撞车
    chunk_id = "web_" + hashlib.sha1((url or snippet).encode("utf-8")).hexdigest()[:16]
    entity = {
        "chunk_id": chunk_id,
        "content": snippet,
        "title": title,
        "parent_title": "",  # 网络文档没有章节层级
        "part": None,
        "file_title": title or url,  # 用标题/链接充当来源名，供答案溯源展示
        "item_name": "",  # 网络结果不属于某个已确认商品主体
        "score": score,
    }
    return {
        # ↓ 与 utils/milvus_utils._normalize_hits 同形：node_rrf 的 doc.get("entity") 可直接取
        "id": chunk_id,
        "chunk_id": chunk_id,
        "distance": score,
        "entity": entity,
        # ↓ 课程既有契约字段：node_rerank._step_1_merge_multi_source_docs 在读
        "title": title,
        "url": url,
        "snippet": snippet,
        # ↓ 三路统一的语义字段
        "content": snippet,
        "source": "web",
        "score": score,
        "item_name": "",
    }


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        执行网络搜索并写入 state["web_search_docs"]。
        任何异常/无结果都只返回空列表：本路是并行分支之一，返回空不会让 node_rrf 崩，
        抛异常则会让整张检索图失败。
        """
        query = state.get("rewritten_query") or state.get("original_query") or ""
        docs: list = []

        try:
            if not query:
                logger.warning("网络搜索：问题为空，跳过")
            else:
                engine = (mcp_config.web_search_engine or ENGINE_TAVILY).strip().lower()
                if engine == ENGINE_MCP:
                    docs = self._search_by_mcp(query)
                else:
                    docs = self._search_by_tavily(query)
                logger.info(f"网络搜索[{engine}]结果: {serialize_json(docs, ensure_ascii=False)}")
        except Exception as e:
            # 兜住一切：网络不通、配额超限、响应结构变化，都降级为空结果
            logger.exception(f"网络搜索失败，降级为空结果: {e}")
            docs = []
        finally:
            # 打点：done 必须执行（含失败路径），否则并行时这一路一直挂在 running，前端进度不更新
            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))

        return {"web_search_docs": docs}

    def _search_by_tavily(self, query: str) -> list:
        """
        Tavily 搜索（本机默认引擎）。

        参数按已安装的 tavily-python 0.8.5 真实签名核对：
        TavilyClient.search(query, search_depth=None, topic=None, max_results=None,
                            include_answer=None, include_raw_content=None, timeout=60, ...)
        返回 dict：{"query", "results": [{"title","url","content","score",...}], "answer", ...}

        :param query: 搜索词（改写后的问题）
        :return: 规整后的文档列表（失败/为空 → []）
        """
        client = _get_tavily_client()
        if client is None:
            return []

        resp = client.search(
            query=query,
            max_results=WEB_SEARCH_TOP_K,  # 对应 MCP 分支的 count
            search_depth="basic",  # basic 最快；advanced 更准但更慢更贵，检索链路里先用 basic
            topic="general",
            include_answer=False,  # 答案由本地 rerank + 生成节点产出，这里只要原始片段
            include_raw_content=False,  # 整页正文太大，会污染 prompt 与 Mongo
            timeout=WEB_SEARCH_TIMEOUT,
        )

        results = (resp or {}).get("results") or []
        docs: list = []
        seen: set = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            snippet = (item.get("content") or "").strip()  # Tavily 的摘要字段名是 content
            if not snippet:
                continue  # 无摘要的结果对 RAG 没有价值，直接丢弃
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            score = item.get("score")
            doc = _build_web_doc(title=title, url=url, snippet=snippet,
                                 score=round(float(score), 4) if isinstance(score, (int, float)) else None)
            if doc["chunk_id"] in seen:  # 同一 URL 去重
                continue
            seen.add(doc["chunk_id"])
            docs.append(doc)

        docs.sort(key=lambda d: (d.get("score") is not None, d.get("score") or 0.0), reverse=True)
        return docs[:WEB_SEARCH_TOP_K]

    def _search_by_mcp(self, query: str) -> list:
        """
        阿里云百炼 DashScope MCP 搜索（课程原方案，保留分支代码，由 web_search_engine=mcp 启用）。
        注意：本机该 MCP key 实测 401，默认不走这条分支。

        :param query: 搜索词
        :return: 规整后的文档列表（失败/为空 → []）
        """
        import asyncio

        result = asyncio.run(self._mcp_call(query))
        if not result:
            return []

        pages = json.loads(result.content[0].text).get("pages") or []
        docs: list = []
        seen: set = set()
        for item in pages:
            snippet = (item.get("snippet") or "").strip()
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            if not snippet:
                continue
            doc = _build_web_doc(title=title, url=url, snippet=snippet)
            if doc["chunk_id"] in seen:
                continue
            seen.add(doc["chunk_id"])
            docs.append(doc)
        return docs

    async def _mcp_call(self, query):
        # MCPServerStreamableHttp 发起一个请求（惰性 import：默认 tavily 引擎不依赖 openai-agents）
        from agents.mcp import MCPServerStreamableHttp

        search_mcp = MCPServerStreamableHttp(
            name="search_mcp",  # 要使用的插件名称(可自定义) 比如我的搜索服务
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": 10,
            },
            cache_tools_list=True,  # 第1次请求,后续缓存不重复请求(服务器上有哪些工具)
            max_retry_attempts=3,  # 网络异常时重试次数
        )

        try:
            await search_mcp.connect()  # 链接
            result = await search_mcp.call_tool(
                tool_name="bailian_web_search",  # 使用的插件名称(不能错)
                arguments={"query": query, "count": WEB_SEARCH_TOP_K},
            )
            return result
        finally:
            await search_mcp.cleanup()


if __name__ == "__main__":
    init_state = {
        "session_id": "test_web_search_001",
        "is_stream": False,
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
    }

    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(serialize_json(result, indent=4))
