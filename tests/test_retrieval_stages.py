"""Tests for the two-stage retrieval flow (build_candidates + rerank_candidates)."""
from __future__ import annotations

import numpy as np

from finrag import models
from finrag.chunking import ChunkResult
from finrag.retrieval import build_candidates, rerank_candidates, retrieve_and_rerank


class _Doc:
    def __init__(self, doc_id: str, text: str = ""):
        self.metadata = {"id": doc_id}
        self.page_content = text


class _FakeVectorstore:
    def __init__(self, ids):
        self.ids = ids

    def similarity_search(self, q, k=10):
        return [_Doc(i) for i in self.ids[:k]]

    def max_marginal_relevance_search(self, q, k=10, fetch_k=30, lambda_mult=0.7):
        return [_Doc(i) for i in self.ids[:k]]


class _FakeBM25Retriever:
    def __init__(self, ids):
        self.ids = ids

    def invoke(self, q):
        return [_Doc(i, f"bm25 text for {i}") for i in self.ids]


class _FakeOkapi:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokens):
        return np.asarray(self.scores, dtype=np.float64)


class _FakeReranker:
    """Scores by reverse doc-id order so rerank ranking != RRF ranking."""

    def predict(self, pairs, batch_size=8, show_progress_bar=False):
        return np.asarray([float(-ord(text[0])) for _, text in pairs], dtype=np.float32)


def _chunk_result():
    # Three chunks: two from doc "a" (one per chunk), one from doc "b".
    return ChunkResult(
        texts=["alpha chunk", "alpha chunk two", "beta chunk"],
        ids=["0_a_chunk0", "1_a_chunk1", "2_b_chunk0"],
        original_ids=["a", "a", "b"],
        lc_docs=[],
    )


def test_build_candidates_returns_ids_and_best_chunks():
    vs = _FakeVectorstore(["a", "b"])
    bm25 = _FakeBM25Retriever(["b"])
    # BM25Okapi scores over the 3 chunks: doc "a" best chunk = "alpha chunk two"
    okapi = _FakeOkapi([1.0, 3.0, 2.0])

    cands = build_candidates(
        vs, bm25, okapi, _chunk_result(), "query",
        dataset_type="tabular", fetch_k=10, rerank_top_n=5, llm=None, mmr_lambda=0.0,
    )
    as_dict = dict(cands)
    assert set(as_dict) == {"a", "b"}
    assert as_dict["a"] == "alpha chunk two"  # highest-scoring chunk wins
    # RRF order: "b" appears in both dense and BM25 lists, so it ranks first
    assert cands[0][0] == "b"


def test_build_candidates_respects_rerank_top_n_and_fallback():
    vs = _FakeVectorstore(["a", "b"])
    bm25 = _FakeBM25Retriever(["b"])
    okapi = _FakeOkapi([0.0, 0.0, 0.0])  # no okapi signal anywhere

    cands = build_candidates(
        vs, bm25, okapi, _chunk_result(), "query",
        dataset_type="tabular", fetch_k=10, rerank_top_n=1, llm=None, mmr_lambda=0.0,
    )
    assert len(cands) == 1
    # Doc "b" has zero okapi scores -> falls back to BM25 page content
    _, text = cands[0]
    assert text in ("bm25 text for a", "bm25 text for b", "alpha chunk", "beta chunk")


def test_rerank_candidates_sorts_and_truncates():
    cands = [("a", "zeta"), ("b", "alpha"), ("c", "beta")]
    ranked = rerank_candidates(_FakeReranker(), "q", cands, k=2)
    # Scores: zeta -> -122, alpha -> -97, beta -> -98; descending = alpha, beta
    assert ranked == [("b", -97.0), ("c", -98.0)]


def test_rerank_candidates_empty():
    assert rerank_candidates(_FakeReranker(), "q", [], k=5) == []


def test_retrieve_and_rerank_compat_wrapper():
    vs = _FakeVectorstore(["a", "b"])
    bm25 = _FakeBM25Retriever(["b"])
    okapi = _FakeOkapi([1.0, 3.0, 2.0])

    ranked = retrieve_and_rerank(
        vs, bm25, okapi, _chunk_result(), "query", _FakeReranker(),
        dataset_type="tabular", fetch_k=10, rerank_top_n=5, k=10, llm=None,
        mmr_lambda=0.0,
    )
    ids = [doc_id for doc_id, _ in ranked]
    assert set(ids) == {"a", "b"}


def test_offload_embedding_model_noop_when_not_cached():
    # Must not raise for a model that was never loaded.
    models.offload_embedding_model("no-such-model/never-loaded")
