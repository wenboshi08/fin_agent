"""Retrieval pipeline (LangChain-native): Dense + BM25 -> RRF -> rerank.

Dense retrieval uses langchain_chroma.Chroma's MMR for passage datasets and
plain similarity for tabular datasets, matching the upstream reference.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi

from .chunking import ChunkResult

logger = logging.getLogger(__name__)


def build_bm25_retriever(lc_docs, fetch_k: int) -> BM25Retriever:
    return BM25Retriever.from_documents(lc_docs, k=fetch_k)


def build_bm25_okapi(texts: list[str]) -> BM25Okapi:
    return BM25Okapi([t.lower().split() for t in texts])


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def expand_query(query: str, llm: Optional[object]) -> list[str]:
    """Use an LLM to create alternative query phrasings. Falls back on failure."""
    if llm is None:
        return [query]
    prompt = (
        "Generate 3 alternative search queries for the following financial question. "
        "Output only the queries, one per line, no numbering or explanation.\n\n"
        f"Original: {query}"
    )
    try:
        response = llm.invoke(prompt)
        variants = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip() and line.strip() != query
        ][:3]
        queries = [query] + variants
        logger.debug("MultiQuery expanded to %d variants.", len(queries))
        return queries
    except Exception as exc:
        logger.warning("MultiQuery expansion failed (%s) — using original query.", exc)
        return [query]


def retrieve_and_rerank(
    vectorstore,
    bm25_retriever: BM25Retriever,
    bm25_okapi: BM25Okapi,
    chunk_result: ChunkResult,
    query: str,
    reranker,
    *,
    dataset_type: str = "passage",
    fetch_k: int = 75,
    rerank_top_n: int = 30,
    k: int = 10,
    llm: Optional[object] = None,
    mmr_lambda: Optional[float] = 0.7,
) -> list[tuple[str, float]]:
    """Full per-query retrieval: dense + BM25 -> RRF -> best chunk -> rerank."""
    queries = expand_query(query, llm) if llm is not None else [query]

    # ── Dense retrieval per query variant ────────────────────────────────
    dense_ranked: list[list[str]] = []
    for q in queries:
        if dataset_type == "passage" and mmr_lambda is not None and mmr_lambda > 0:
            docs = vectorstore.max_marginal_relevance_search(
                q, k=fetch_k, fetch_k=fetch_k * 3, lambda_mult=mmr_lambda
            )
        else:
            docs = vectorstore.similarity_search(q, k=fetch_k)
        dense_ranked.append([d.metadata["id"] for d in docs])

    # ── BM25 retrieval on original query ────────────────────────────────
    bm25_docs = bm25_retriever.invoke(query)
    bm25_ranked = [d.metadata["id"] for d in bm25_docs]

    # ── RRF fusion ──────────────────────────────────────────────────────
    fused = rrf_fuse(dense_ranked + [bm25_ranked], k=60)
    candidate_ids: list[str] = []
    for doc_id, _ in fused:
        if doc_id not in candidate_ids:
            candidate_ids.append(doc_id)
        if len(candidate_ids) >= rerank_top_n:
            break

    if not candidate_ids:
        return []

    # ── Best chunk per candidate doc using BM25Okapi scores ─────────────
    bm25_scores = bm25_okapi.get_scores(query.lower().split())
    candidate_set = set(candidate_ids)
    best_chunk: dict[str, tuple[float, str]] = {}
    for chunk_text, orig_id, score in zip(
        chunk_result.texts, chunk_result.original_ids, bm25_scores
    ):
        if orig_id not in candidate_set:
            continue
        s = float(score)
        if orig_id not in best_chunk or s > best_chunk[orig_id][0]:
            best_chunk[orig_id] = (s, chunk_text)

    # Build fallback content from BM25 docs and chunk_result
    seen_content: dict[str, str] = {d.metadata["id"]: d.page_content for d in bm25_docs}
    chunk_by_doc: dict[str, str] = {}
    for text, orig_id in zip(chunk_result.texts, chunk_result.original_ids):
        if orig_id not in chunk_by_doc or len(text) > len(chunk_by_doc[orig_id]):
            chunk_by_doc[orig_id] = text

    for oid in candidate_ids:
        if oid not in best_chunk:
            fallback = seen_content.get(oid) or chunk_by_doc.get(oid, "")
            best_chunk[oid] = (0.0, fallback)

    # ── Cross-encoder rerank ────────────────────────────────────────────
    pairs = [(query, best_chunk[oid][1]) for oid in candidate_ids]
    scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    ranked = sorted(
        zip(candidate_ids, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:k]
