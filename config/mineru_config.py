#knowledge_base\config\mineru_config.py
# 区分 test（测试环境）和 prod（Production / 生产环境）后缀，是软件开发中非常经典且重要的环境隔离最佳实践。
from dataclasses import dataclass
import os
from dotenv import load_dotenv

#1.加载配置文件
load_dotenv(override=True)

@dataclass
class MinerUConfig:

    base_url_prod: str
    api_token_prod: str
    base_url_test: str
    api_token_test: str
    # 文档解析引擎：pymupdf4llm（本地，默认）| mineru（在线，需要有效 token）
    pdf_engine: str

mineru_config = MinerUConfig(
    base_url_prod=os.getenv("MINERU_BASE_URL", ""),
    api_token_prod=os.getenv("MINERU_API_TOKEN", ""),
    base_url_test = "test",
    api_token_test = "test",
    pdf_engine=os.getenv("PDF_TO_MD_ENGINE", "pymupdf4llm")
)