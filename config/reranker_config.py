# config/reranker_config.py

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class RerankerConfig:
    text_rerank_api_key: str # DashScope API Key
    text_rerank_model: str # 模型名称
    text_rerank_instruct: str # 是否使用指令
    # 本机无可用 rerank API 额度，走本地 FlagReranker
    rerank_mode: str
    reranker_path: str
    reranker_device: str
    reranker_fp16: bool

reranker_config = RerankerConfig(
    text_rerank_api_key=os.getenv("MCP_API_KEY"),
    text_rerank_model=os.getenv("TEXT_RERANK_MODEL"),
    text_rerank_instruct=os.getenv("TEXT_RERANK_INSTRUCT"),
    rerank_mode=os.getenv("RERANK_MODE", "local"),
    reranker_path=os.getenv("BGE_RERANKER_LARGE"),
    reranker_device=os.getenv("BGE_RERANKER_DEVICE", "cpu"),
    reranker_fp16=os.getenv("BGE_RERANKER_FP16", "0") in ("1", "True", "true")
)