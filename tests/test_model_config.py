"""Tests for model defaults, aliases, and the new model adapters."""
from __future__ import annotations

import types

import numpy as np

from finrag import config, models
from finrag.models import E5InstructEmbeddings, FlagLLMRerankerAdapter

# ── Config consistency ──────────────────────────────────────────────────────


def test_default_embedding_model_is_a_choice():
    assert config.DEFAULT_EMBEDDING_MODEL in config.EMBEDDING_MODEL_CHOICES.values()
    assert config.EMBEDDING_MODEL_CHOICES["e5-mistral"] == config.DEFAULT_EMBEDDING_MODEL


def test_default_reranker_model_is_a_choice():
    assert config.DEFAULT_RERANKER_MODEL in config.RERANKER_MODEL_CHOICES.values()
    assert config.RERANKER_MODEL_CHOICES["bge-gemma"] == config.DEFAULT_RERANKER_MODEL


def test_default_models_have_required_specs():
    # Instruction-based embedding default must have a usage spec.
    assert config.DEFAULT_EMBEDDING_MODEL in config.INSTRUCT_EMBEDDING_SPECS
    spec = config.INSTRUCT_EMBEDDING_SPECS[config.DEFAULT_EMBEDDING_MODEL]
    assert spec["query_instruction"]
    # LLM reranker default must be routed through FlagLLMReranker.
    assert config.DEFAULT_RERANKER_MODEL in config.LLM_RERANKER_MODELS


def test_legacy_aliases_still_resolve():
    assert "finlang" in config.EMBEDDING_MODEL_CHOICES
    assert "bge-m3" in config.RERANKER_MODEL_CHOICES


# ── E5InstructEmbeddings ────────────────────────────────────────────────────


class _FakeSTModel:
    """Minimal stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self):
        self.calls: list[tuple[list[str], bool, bool]] = []

    def encode(self, texts, batch_size=1, normalize_embeddings=False,
               show_progress_bar=False, convert_to_numpy=True):
        self.calls.append((list(texts), normalize_embeddings, show_progress_bar))
        return np.ones((len(texts), 4), dtype=np.float32)


def test_e5_instruct_query_gets_instruction_prefix():
    fake = _FakeSTModel()
    emb = E5InstructEmbeddings(fake, query_instruction="TASK", batch_size=2)
    vec = emb.embed_query("What was revenue?")
    texts, normalized, _ = fake.calls[-1]
    assert texts == ["Instruct: TASK\nQuery: What was revenue?"]
    assert normalized is True
    assert vec == [1.0, 1.0, 1.0, 1.0]


def test_e5_instruct_documents_have_no_prefix():
    fake = _FakeSTModel()
    emb = E5InstructEmbeddings(fake, query_instruction="TASK")
    out = emb.embed_documents(["doc one", "doc two"])
    texts, normalized, _ = fake.calls[-1]
    assert texts == ["doc one", "doc two"]
    assert normalized is True
    assert len(out) == 2


# ── FlagLLMRerankerAdapter ──────────────────────────────────────────────────


class _FakeLLMReranker:
    def __init__(self, scalar=False):
        self.scalar = scalar
        self.kwargs = None

    def compute_score(self, pairs, batch_size=256, max_length=512):
        self.kwargs = {"batch_size": batch_size, "max_length": max_length}
        if self.scalar:
            return 1.5
        return [float(i) for i in range(len(pairs))]


def test_llm_reranker_adapter_predict_list():
    fake = _FakeLLMReranker()
    adapter = FlagLLMRerankerAdapter(fake, "model-x", max_length=256)
    scores = adapter.predict([("q", "p1"), ("q", "p2")], batch_size=4)
    assert isinstance(scores, np.ndarray)
    assert scores.tolist() == [0.0, 1.0]
    assert fake.kwargs == {"batch_size": 4, "max_length": 256}


def test_llm_reranker_adapter_handles_scalar_and_empty():
    fake = _FakeLLMReranker(scalar=True)
    adapter = FlagLLMRerankerAdapter(fake, "model-x")
    assert adapter.predict([("q", "p")]).tolist() == [1.5]
    assert adapter.predict([]).size == 0


# ── GPU/CPU offload lifecycle ───────────────────────────────────────────────


class _MovableModule:
    def __init__(self):
        self.device = None
        self.moves: list[str] = []

    def to(self, device):
        self.device = device
        self.moves.append(str(device))


def test_offload_embedding_model_and_auto_relocate():
    fake = _MovableModule()
    emb = E5InstructEmbeddings(fake, query_instruction="TASK")
    name = "unit-test/fake-embedding-model"
    models._embedding_cache[name] = emb
    try:
        models.offload_embedding_model(name)
        assert fake.device == "cpu"
        # Next access moves the cached model back onto the best device.
        again = models.get_embedding_model(name)
        assert again is emb
        assert fake.moves[-1] == models.get_device()
    finally:
        models._embedding_cache.pop(name, None)


def test_offload_reranker_moves_cached_reranker_to_cpu():
    fake = _MovableModule()
    adapter = FlagLLMRerankerAdapter(types.SimpleNamespace(model=fake), "model-x")
    prev = models._reranker
    models._reranker = adapter
    try:
        models.offload_reranker()
        assert fake.device == "cpu"
        # get_reranker cache hit relocates it back.
        models.get_reranker("model-x")
        assert fake.moves[-1] == models.get_device()
    finally:
        models._reranker = prev


def test_offload_reranker_noop_when_not_loaded():
    prev = models._reranker
    models._reranker = None
    try:
        models.offload_reranker()  # must not raise
    finally:
        models._reranker = prev
