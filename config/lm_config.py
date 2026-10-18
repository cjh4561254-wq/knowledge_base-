# knowledge_base\config\lm_config.py

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class LLMConfig:
    base_url: str
    api_key : str
    vl_model: str
    llm_model: str
    llm_temperature: float
    item_model: str
    # 视觉模型走本地 Ollama 的 OpenAI 兼容端点，与文本 LLM 的 base_url 分开
    vl_base_url: str
    vl_api_key: str
# 读配置
lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    vl_model=os.getenv("VL_MODEL"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE")),
    item_model=os.getenv("ITEM_MODEL"),
    vl_base_url=os.getenv("VL_BASE_URL", "http://localhost:11434/v1"),
    vl_api_key=os.getenv("VL_API_KEY", "ollama")
)
