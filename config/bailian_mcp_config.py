# config/bailian_mcp_config.py

# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(override=True)


# 定义mcp的服务配置
@dataclass
class McpConfig:
    mcp_base_url: str
    api_key : str
    # 搜索引擎：tavily（本机可用）| mcp（DashScope，实测 key 已 401）
    web_search_engine: str
    tavily_api_key: str

mcp_config = McpConfig(
    mcp_base_url=os.getenv("MCP_DASHSCOPE_BASE_URL"),
    api_key=os.getenv("MCP_API_KEY"),# 硅基流动
    web_search_engine=os.getenv("WEB_SEARCH_ENGINE", "tavily"),
    tavily_api_key=os.getenv("TAVILY_API_KEY", "")
)