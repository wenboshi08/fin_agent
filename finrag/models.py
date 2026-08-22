"""Lazy model singletons: LangChain embeddings, reranker, DeepSeek/Qwen ChatOpenAI."""
from __future__ import annotations

import logging
import os
from typing import Optional

from . import config

logger = logging.getLogger(__name__)


def get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


_embedding_cache: dict[str, object] = {}


def get_embedding_model(model_name: str | None = None):
    """Return a cached LangChain HuggingFaceEmbeddings instance."""
    from langchain_huggingface import HuggingFaceEmbeddings

    name = model_name or config.DEFAULT_EMBEDDING_MODEL
    if name in _embedding_cache:
        return _embedding_cache[name]

    device = get_device()
    logger.info("Loading embedding model %s on device=%s ...", name, device)
    emb = HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    )
    _embedding_cache[name] = emb
    return emb


_reranker = None


def get_reranker(model_name: str | None = None):
    """Return a CrossEncoder reranker singleton."""
    global _reranker

    from sentence_transformers import CrossEncoder

    name = model_name or config.DEFAULT_RERANKER_MODEL
    if _reranker is None or getattr(_reranker, "model_name", None) != name:
        logger.info("Loading reranker %s ...", name)
        _reranker = CrossEncoder(name, device=get_device(), max_length=512)
        _reranker.model_name = name  # type: ignore[attr-defined]
    return _reranker


_PROVIDER_CONFIG = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_model": config.DEEPSEEK_DEFAULT_MODEL,
        "default_base_url": config.DEEPSEEK_BASE_URL,
    },
    "qwen": {
        "api_key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "default_model": config.QWEN_DEFAULT_MODEL,
        "default_base_url": config.QWEN_BASE_URL,
    },
}


def _resolve_provider(provider: str | None):
    provider = (provider or config.DEFAULT_LLM_PROVIDER).lower()
    if provider not in _PROVIDER_CONFIG:
        raise ValueError(f"Unknown provider '{provider}'. Choose from {sorted(_PROVIDER_CONFIG)}")
    cfg = _PROVIDER_CONFIG[provider]
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        logger.warning(
            "%s not set — LLM disabled. Add it to .env to enable generation/MultiQuery.",
            cfg["api_key_env"],
        )
        return None
    model = config.DEFAULT_LLM_MODEL or cfg["default_model"]
    base_url = os.getenv(cfg["base_url_env"], cfg["default_base_url"])
    return provider, model, api_key, base_url


def get_llm(provider: str | None = None, model: str | None = None) -> Optional[object]:
    """Return a ChatOpenAI client for DeepSeek or Qwen, or None without API key."""
    from langchain_openai import ChatOpenAI

    resolved = _resolve_provider(provider)
    if resolved is None:
        return None
    provider_name, default_model, api_key, base_url = resolved
    model_name = model or default_model
    logger.info("Loading LLM %s/%s ...", provider_name, model_name)
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=512,
    )


def get_eval_llm(provider: str | None = None, model: str | None = None):
    """Return a rate-limited judge LLM for RAGAS evaluation."""
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_openai import ChatOpenAI

    resolved = _resolve_provider(provider)
    if resolved is None:
        return None
    provider_name, default_model, api_key, base_url = resolved
    model_name = model or default_model
    # Conservative limiter so RAGAS's per-sample judge calls don't hit API limits.
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=2.0,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=1024,
        rate_limiter=rate_limiter,
    )
