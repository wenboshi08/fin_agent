"""High-level pipeline helpers (LangChain-based)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .chunking import split_documents
from .data import load_jsonl, load_qrels, make_corpus_lookup
from .evaluation import compute_ndcg
from .generation import CitedSource, GenerationResult, generate_answer
from .models import (
    get_embedding_model,
    get_llm,
    get_reranker,
    offload_embedding_model,
    offload_reranker,
)
from .retrieval import (
    build_bm25_okapi,
    build_bm25_retriever,
    build_candidates,
    rerank_candidates,
)
from .vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def dataset_paths(dataset_name: str) -> dict[str, Path]:
    cfg = config.DATASET_CONFIGS[dataset_name]
    return {
        "corpus": config.DATASET_DIR / cfg.corpus_file,
        "queries": config.DATASET_DIR / cfg.query_file,
        "qrels": config.DATASET_DIR / cfg.qrels_file,
    }


@dataclass
class BenchmarkResult:
    name: str
    ndcg: float
    num_queries: int
    num_chunks: int
    elapsed_sec: float
    results: dict[str, dict[str, float]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def load_dataset(dataset_name: str):
    paths = dataset_paths(dataset_name)
    corpus = load_jsonl(paths["corpus"])
    queries = load_jsonl(paths["queries"])
    qrels = load_qrels(paths["qrels"])
    return corpus, queries, qrels


def prepare_retriever(
    dataset_name: str,
    *,
    chunk_strategy: str = "dataset-aware",
    embedding_model: str | None = None,
    force_rebuild: bool = False,
):
    cfg = config.DATASET_CONFIGS[dataset_name]
    corpus, queries, qrels = load_dataset(dataset_name)

    chunks = split_documents(
        corpus,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        dataset_type=cfg.dataset_type,
        strategy=chunk_strategy,
    )

    embedding_model_name = embedding_model or config.DEFAULT_EMBEDDING_MODEL
    embedding = get_embedding_model(embedding_model_name)
    vectorstore = get_vectorstore(
        dataset_name,
        chunks,
        embedding,
        force_rebuild=force_rebuild,
        model_name=embedding_model_name,
    )
    bm25_retriever = build_bm25_retriever(chunks.lc_docs, fetch_k=cfg.fetch_k)
    bm25_okapi = build_bm25_okapi(chunks.texts)

    return {
        "dataset_name": dataset_name,
        "cfg": cfg,
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
        "chunks": chunks,
        "vectorstore": vectorstore,
        "bm25_retriever": bm25_retriever,
        "bm25_okapi": bm25_okapi,
        "corpus_lookup": make_corpus_lookup(corpus),
    }


def run_dataset_benchmark(
    dataset_name: str,
    *,
    top_k: int = 10,
    use_multiquery: bool = True,
    force_rebuild: bool = False,
    provider: str = "deepseek",
    model: str | None = None,
    chunk_strategy: str = "dataset-aware",
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    mmr_lambda: float = 0.7,
) -> BenchmarkResult:
    start = time.time()
    data = prepare_retriever(
        dataset_name,
        chunk_strategy=chunk_strategy,
        embedding_model=embedding_model,
        force_rebuild=force_rebuild,
    )
    cfg = data["cfg"]
    llm = get_llm(provider, model) if use_multiquery else None

    results: dict[str, dict[str, float]] = {}
    errors: list[str] = []

    # ── Pass 1: candidate generation (embedding model resident) ──────────
    # A reranker cached from a previous dataset run must leave the GPU first.
    offload_reranker()
    candidates: dict[str, list[tuple[str, str]]] = {}
    for q in data["queries"]:
        qid = q["_id"]
        try:
            candidates[qid] = build_candidates(
                data["vectorstore"],
                data["bm25_retriever"],
                data["bm25_okapi"],
                data["chunks"],
                q["text"],
                dataset_type=cfg.dataset_type,
                fetch_k=cfg.fetch_k,
                rerank_top_n=cfg.rerank_top_n,
                llm=llm,
                mmr_lambda=mmr_lambda,
            )
        except Exception as exc:
            errors.append(f"{qid}: {exc}")
            logger.exception("Candidate retrieval failed for query %s", qid)

    # ── Free GPU memory, then load the reranker ──────────────────────────
    offload_embedding_model(embedding_model)
    reranker = get_reranker(reranker_model)

    # ── Pass 2: rerank candidates ────────────────────────────────────────
    for q in data["queries"]:
        qid = q["_id"]
        if qid not in candidates:
            continue
        try:
            ranked = rerank_candidates(reranker, q["text"], candidates[qid], k=top_k)
            results[qid] = {doc_id: float(score) for doc_id, score in ranked}
        except Exception as exc:
            errors.append(f"{qid}: {exc}")
            logger.exception("Reranking failed for query %s", qid)

    ndcg = compute_ndcg(data["qrels"], results, k=top_k)
    elapsed = time.time() - start
    return BenchmarkResult(
        name=dataset_name,
        ndcg=ndcg,
        num_queries=len(data["queries"]),
        num_chunks=len(data["chunks"].texts),
        elapsed_sec=elapsed,
        results=results,
        errors=errors,
    )


def run_rag_query(
    query: str,
    dataset_name: str,
    *,
    top_k: int = 5,
    use_multiquery: bool = True,
    force_rebuild: bool = False,
    provider: str = "deepseek",
    model: str | None = None,
    chunk_strategy: str = "dataset-aware",
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    mmr_lambda: float = 0.7,
) -> GenerationResult:
    data = prepare_retriever(
        dataset_name,
        chunk_strategy=chunk_strategy,
        embedding_model=embedding_model,
        force_rebuild=force_rebuild,
    )
    cfg = data["cfg"]
    llm = get_llm(provider, model)

    # ── Pass 1: candidate generation (embedding model resident) ──────────
    offload_reranker()
    candidates = build_candidates(
        data["vectorstore"],
        data["bm25_retriever"],
        data["bm25_okapi"],
        data["chunks"],
        query,
        dataset_type=cfg.dataset_type,
        fetch_k=cfg.fetch_k,
        rerank_top_n=cfg.rerank_top_n,
        llm=llm if use_multiquery else None,
        mmr_lambda=mmr_lambda,
    )

    # ── Free GPU memory, then rerank ─────────────────────────────────────
    offload_embedding_model(embedding_model)
    reranker = get_reranker(reranker_model)
    retrieved = rerank_candidates(reranker, query, candidates, k=top_k)

    if llm is None:
        return GenerationResult(
            query=query,
            answer="No LLM configured (set DEEPSEEK_API_KEY or QWEN_API_KEY). Retrieved docs:\n"
            + "\n".join(doc_id for doc_id, _ in retrieved),
            sources=[
                CitedSource(
                    doc_id, data["corpus_lookup"].get(doc_id, {}).get("title", ""), score, ""
                )
                for doc_id, score in retrieved[:top_k]
            ],
            model="none",
            multiquery=use_multiquery,
        )
    return generate_answer(
        query,
        retrieved,
        data["corpus_lookup"],
        llm,
        top_k=top_k,
        multiquery=use_multiquery,
        model_name=getattr(llm, "model_name", "unknown"),
    )
