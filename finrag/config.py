"""Project paths and per-dataset configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
CHROMA_DIR = BASE_DIR / "chroma_stores"
RESULTS_DIR = BASE_DIR / "results"
CONFIGS_DIR = BASE_DIR / "configs"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    corpus_file: str
    query_file: str
    qrels_file: str
    chunk_size: int
    chunk_overlap: int
    dataset_type: str  # "passage" | "tabular"
    fetch_k: int
    rerank_top_n: int


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "financebench": DatasetConfig(
        name="financebench",
        corpus_file="financebench_corpus.jsonl/corpus.jsonl",
        query_file="financebench_queries.jsonl/queries.jsonl",
        qrels_file="FinanceBench_qrels.tsv",
        chunk_size=512,
        chunk_overlap=64,
        dataset_type="passage",
        fetch_k=75,
        rerank_top_n=30,
    ),
    "finder": DatasetConfig(
        name="finder",
        corpus_file="finder_corpus.jsonl/corpus.jsonl",
        query_file="finder_queries.jsonl/queries.jsonl",
        qrels_file="FinDER_qrels.tsv",
        chunk_size=512,
        chunk_overlap=64,
        dataset_type="passage",
        fetch_k=75,
        rerank_top_n=30,
    ),
    "finqabench": DatasetConfig(
        name="finqabench",
        corpus_file="finqabench_corpus.jsonl/corpus.jsonl",
        query_file="finqabench_queries.jsonl/queries.jsonl",
        qrels_file="FinQABench_qrels.tsv",
        chunk_size=1024,
        chunk_overlap=128,
        dataset_type="passage",
        fetch_k=75,
        rerank_top_n=30,
    ),
    "finqa": DatasetConfig(
        name="finqa",
        corpus_file="finqa_corpus.jsonl/corpus.jsonl",
        query_file="finqa_queries.jsonl/queries.jsonl",
        qrels_file="FinQA_qrels.tsv",
        chunk_size=1024,
        chunk_overlap=128,
        dataset_type="tabular",
        fetch_k=40,
        rerank_top_n=20,
    ),
    "tatqa": DatasetConfig(
        name="tatqa",
        corpus_file="tatqa_corpus.jsonl/corpus.jsonl",
        query_file="tatqa_queries.jsonl/queries.jsonl",
        qrels_file="TATQA_qrels.tsv",
        chunk_size=1024,
        chunk_overlap=128,
        dataset_type="tabular",
        fetch_k=50,
        rerank_top_n=25,
    ),
    "convfinqa": DatasetConfig(
        name="convfinqa",
        corpus_file="convfinqa_corpus.jsonl/corpus.jsonl",
        query_file="convfinqa_queries.jsonl/queries.jsonl",
        qrels_file="ConvFinQA_qrels.tsv",
        chunk_size=1024,
        chunk_overlap=128,
        dataset_type="tabular",
        fetch_k=50,
        rerank_top_n=25,
    ),
    "multiheirtt": DatasetConfig(
        name="multiheirtt",
        corpus_file="multiheirtt_corpus.jsonl/corpus.jsonl",
        query_file="multiheirtt_queries.jsonl/queries.jsonl",
        qrels_file="MultiHeirtt_qrels.tsv",
        chunk_size=1024,
        chunk_overlap=128,
        dataset_type="tabular",
        fetch_k=40,
        rerank_top_n=20,
    ),
}

# ── Model defaults ──────────────────────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "FinLang/finance-embeddings-investopedia"
EMBEDDING_MODEL_CHOICES = {
    "finlang": DEFAULT_EMBEDDING_MODEL,
    "bge-m3": "BAAI/bge-m3",
    "all-mpnet": "sentence-transformers/all-mpnet-base-v2",
}

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_CHOICES = {
    "bge": DEFAULT_RERANKER_MODEL,
    "mini": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "electra": "cross-encoder/ms-marco-electra-base",
}

# ── LLM (DeepSeek / Qwen, OpenAI-compatible) ───────────────────────────────
DEEPSEEK_DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
QWEN_DEFAULT_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

DEFAULT_LLM_PROVIDER = os.getenv("FINAGENT_LLM_PROVIDER", "deepseek")
DEFAULT_LLM_MODEL = os.getenv("FINAGENT_LLM_MODEL", "")  # empty = provider default

# ── Defaults from upstream / MVP ────────────────────────────────────────────
DEFAULT_TOP_K = 10
DEFAULT_RAG_TOP_K = 5
DEFAULT_MMR_LAMBDA = 0.7
