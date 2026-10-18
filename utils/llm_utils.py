# utils/llm_utils.py
"""
统一 LLM 客户端工厂（LangChain ChatOpenAI）。

为什么这样写（本机契约）：
1. 本机文本 LLM 已切到 DeepSeek 官方端点（只有 deepseek-flash / deepseek-v4-pro）。
   旧版无条件传 `extra_body={"enable_thinking": False}` —— 那是 SiliconFlow/百炼
   专有参数，DeepSeek 官方对未知参数直接 400。现在按 base_url 主机名条件启用。
2. 旧版不传 timeout/max_retries：API 一旦挂起会把整条 LangGraph 占死
   （检索图并行三路，任何一路挂在 HTTP 上，整个会话的 SSE 进度条就不会结束）。
   现在默认 60s 超时 + 2 次重试，可用环境变量 LLM_TIMEOUT / LLM_MAX_RETRIES 覆盖。
3. 单例缓存 key 从 (model, json_mode) 扩成 (model, base_url, json_mode)：
   文本模型(llm_model)与商品名模型(item_model)乃至将来挂到不同端点的模型
   不会互相串客户端。
4. get_llm_client() 会被并行节点（item_name_confirm / hyde / answer_output）
   并发调用，dict 的「查-建-写」不是原子操作，加锁保证同一 key 只建一个实例。
"""
import os
import threading

from langchain_openai import ChatOpenAI

from config.lm_config import lm_config

_llm_client_cache = {}
# 双检锁：并发首次 get_llm_client 时避免各建一份 client
_cache_lock = threading.Lock()

# 只有这些兼容端点认识 enable_thinking 参数（SiliconFlow / 阿里百炼等）
_THINKING_PARAM_HOSTS = ("siliconflow", "dashscope", "aliyuncs")

# 挂起保护：单次请求超时（秒）与 SDK 层重试次数
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_RETRIES = 2


def _env_num(name: str, default: float) -> float:
    """读环境变量数值，非法值回退默认（绝不让配置错误把 client 工厂打挂）。"""
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


def _extra_body_for(base_url: str | None) -> dict | None:
    """
    按端点决定是否携带 enable_thinking=False。
    DeepSeek 官方端点返回 None（不传 extra_body），避免 400 unknown parameter。
    """
    url = (base_url or "").lower()
    if any(host in url for host in _THINKING_PARAM_HOSTS):
        return {"enable_thinking": False}
    return None


def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取 LangChain ChatOpenAI 客户端实例
    - model: 允许不同节点使用不同模型（如 item_model 走便宜档）
    - json_mode: True 时要求输出 JSON
    """
    m = model or lm_config.llm_model
    base_url = lm_config.base_url
    key = (m, base_url, json_mode)
    client = _llm_client_cache.get(key)
    if client is not None:
        return client

    with _cache_lock:
        # 双检：等锁期间可能已被其他节点建好
        client = _llm_client_cache.get(key)
        if client is None:
            model_kwargs: dict = {}
            if json_mode:
                model_kwargs["response_format"] = {"type": "json_object"}

            client = ChatOpenAI(
                model=m,
                temperature=lm_config.llm_temperature,
                api_key=lm_config.api_key,
                base_url=base_url,
                extra_body=_extra_body_for(base_url),
                model_kwargs=model_kwargs,
                # 超时与重试：API 挂起时快速失败，不拖死整张图
                timeout=_env_num("LLM_TIMEOUT", _DEFAULT_TIMEOUT),
                max_retries=int(_env_num("LLM_MAX_RETRIES", _DEFAULT_MAX_RETRIES)),
            )
            _llm_client_cache[key] = client
    return client
