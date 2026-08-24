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


def _torch_dtype(device: str):
    """Pick a compute dtype suited to the device (bf16 on CUDA, fp32 on CPU)."""
    import torch

    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


class E5InstructEmbeddings:
    """LangChain-compatible embeddings for instruction-based LLM encoders.

    Used by the E5-Mistral family (``intfloat/e5-mistral-7b-instruct`` and the
    finance-adapted Fin-E5 fine-tuned from it): queries are prefixed with a
    task instruction, documents are embedded as-is; pooling is last-token and
    outputs are unit-normalized (the model card's intended usage).
    """

    def __init__(self, model, query_instruction: str, batch_size: int = 4):
        self.model = model
        self.query_instruction = query_instruction
        self.batch_size = batch_size

    def _encode(self, texts: list[str], show_progress: bool):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._encode(list(texts), show_progress=len(texts) > 32)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"Instruct: {self.query_instruction}\nQuery: {text}"
        return self._encode([prefixed], show_progress=False)[0].tolist()


_embedding_cache: dict[str, object] = {}


def _underlying_st_model(emb):
    """Best-effort access to the SentenceTransformer behind an embeddings object."""
    return getattr(emb, "model", None) or getattr(emb, "client", None)


def _relocate_embeddings(emb, device: str) -> None:
    """Move an embeddings object's model onto ``device`` (best effort)."""
    model = _underlying_st_model(emb)
    if model is None:
        return
    try:
        model.to(device)
    except Exception as exc:
        logger.warning("Could not move embedding model to %s: %s", device, exc)


def get_embedding_model(model_name: str | None = None):
    """Return a cached embeddings instance (LangChain-compatible API).

    A cached model that was offloaded to CPU (see :func:`offload_embedding_model`)
    is moved back onto the best available device automatically.
    """
    name = model_name or config.DEFAULT_EMBEDDING_MODEL
    if name in _embedding_cache:
        emb = _embedding_cache[name]
        _relocate_embeddings(emb, get_device())
        return emb

    device = get_device()
    if name in config.INSTRUCT_EMBEDDING_SPECS:
        from sentence_transformers import SentenceTransformer

        spec = config.INSTRUCT_EMBEDDING_SPECS[name]
        dtype = _torch_dtype(device)
        logger.info("Loading LLM embedding model %s on device=%s dtype=%s ...", name, device, dtype)
        model = SentenceTransformer(
            name,
            device=device,
            model_kwargs={"torch_dtype": dtype},
        )
        if spec.get("max_seq_length"):
            model.max_seq_length = int(spec["max_seq_length"])
        emb = E5InstructEmbeddings(
            model,
            query_instruction=spec.get(
                "query_instruction", config.DEFAULT_EMBEDDING_QUERY_INSTRUCTION
            ),
            batch_size=int(
                os.getenv("FINAGENT_EMBED_BATCH_SIZE", str(spec.get("batch_size", 4)))
            ),
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model %s on device=%s ...", name, device)
        emb = HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": int(os.getenv("FINAGENT_EMBED_BATCH_SIZE", "32")),
                "normalize_embeddings": True,
            },
        )
    _embedding_cache[name] = emb
    return emb


def offload_embedding_model(model_name: str | None = None) -> None:
    """Move a cached embedding model to CPU and free GPU memory.

    Called between retrieval stages: the LLM embedding model (~14 GB bf16) and
    the LLM reranker (~5 GB + large logits) do not fit on one consumer GPU at
    the same time. The model stays in the cache and is moved back to the GPU
    automatically on the next :func:`get_embedding_model` call.
    """
    name = model_name or config.DEFAULT_EMBEDDING_MODEL
    emb = _embedding_cache.get(name)
    if emb is None:
        return
    _relocate_embeddings(emb, "cpu")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("Offloaded embedding model %s to CPU; GPU memory released.", name)


class FlagLLMRerankerAdapter:
    """``.predict(pairs, ...)``-compatible adapter for FlagLLMReranker.

    Decoder-only rerankers (e.g. bge-reranker-v2-gemma) score a pair with the
    logit of the 'Yes' token instead of a classification head, so they cannot
    be loaded as sentence-transformers CrossEncoders.
    """

    def __init__(self, reranker, model_name: str, max_length: int = 512):
        self._reranker = reranker
        self.model_name = model_name
        self.max_length = max_length

    def predict(self, pairs, batch_size: int | None = None, show_progress_bar: bool = False):
        import numpy as np

        pairs = list(pairs)
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        scores = self._reranker.compute_score(
            pairs,
            batch_size=batch_size or config.RERANK_PREDICT_BATCH_SIZE,
            max_length=self.max_length,
        )
        if isinstance(scores, (int, float)):
            scores = [scores]
        return np.asarray(scores, dtype=np.float32)


def _reranker_torch_module(reranker):
    """Best-effort access to the torch module behind a cached reranker."""
    if isinstance(reranker, FlagLLMRerankerAdapter):
        return getattr(reranker._reranker, "model", None)
    return getattr(reranker, "model", None)


def _relocate_reranker(reranker, device: str) -> None:
    module = _reranker_torch_module(reranker)
    if module is None:
        return
    try:
        module.to(device)
    except Exception as exc:
        logger.warning("Could not move reranker to %s: %s", device, exc)


def offload_reranker() -> None:
    """Move a cached reranker to CPU and free GPU memory.

    Called before the candidate-generation pass of a subsequent dataset so the
    embedding model and the reranker are never resident on the GPU together.
    """
    if _reranker is None:
        return
    _relocate_reranker(_reranker, "cpu")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info(
        "Offloaded reranker %s to CPU; GPU memory released.",
        getattr(_reranker, "model_name", "unknown"),
    )


_reranker = None


def get_reranker(model_name: str | None = None):
    """Return a reranker singleton exposing ``.predict(pairs, ...) -> scores``.

    A cached reranker that was offloaded to CPU is moved back onto the best
    available device automatically.
    """
    global _reranker

    name = model_name or config.DEFAULT_RERANKER_MODEL
    if _reranker is not None and getattr(_reranker, "model_name", None) == name:
        _relocate_reranker(_reranker, get_device())
        return _reranker

    device = get_device()
    if name in config.LLM_RERANKER_MODELS:
        from FlagEmbedding import FlagLLMReranker

        use_fp16 = device == "cuda"
        logger.info("Loading LLM reranker %s on device=%s fp16=%s ...", name, device, use_fp16)
        reranker = FlagLLMReranker(
            name,
            devices=[device] if device == "cuda" else None,
            use_fp16=use_fp16,
            max_length=config.RERANKER_MAX_LENGTH,
        )
        _reranker = FlagLLMRerankerAdapter(
            reranker, name, max_length=config.RERANKER_MAX_LENGTH
        )
    else:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker %s ...", name)
        _reranker = CrossEncoder(name, device=device, max_length=config.RERANKER_MAX_LENGTH)
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
