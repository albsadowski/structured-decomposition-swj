from functools import cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.globals import set_llm_cache
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.cache import SQLiteCache


@cache
def chat_model(model: str, use_cache: bool = True) -> BaseChatModel:
    kwargs, provider = {}, "openai"
    if use_cache:
        set_llm_cache(SQLiteCache(database_path="./.cache"))
    if model in ["gpt-4o-mini", "gpt-4o"]:
        kwargs["temperature"] = 0.0
    if model == "qwen-3":
        model = "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "deepseek-v3.2":
        model = "accounts/fireworks/models/deepseek-v3p2"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "claude-haiku":
        model = "claude-haiku-4-5-20251001"
        provider = "anthropic"
        kwargs["temperature"] = 0.0
    if model == "claude-sonnet":
        model = "claude-sonnet-4-5-20250929"
        provider = "anthropic"
        kwargs["temperature"] = 0.0
    if model == "gemini-2.5-flash":
        provider = "google_genai"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "gemini-2.5-pro":
        provider = "google_genai"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "kimi-k2":
        model = "accounts/fireworks/models/kimi-k2-instruct-0905"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    return init_chat_model(model, model_provider=provider, **kwargs)
