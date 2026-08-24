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
# Default dense retriever: E5-Mistral-7B (instruction-based, LLM encoder).
# This is the base model that Fin-E5 (FinMTEB, arXiv:2502.10990) was fine-tuned
# from; the Fin-E5 weights are currently access-gated, so we run the base model
# with the same usage pattern. Once access to FinanceMTEB/FinE5 is granted,
# swap the model id below (and in INSTRUCT_EMBEDDING_SPECS) — no other changes.
DEFAULT_EMBEDDING_MODEL = "intfloat/e5-mistral-7b-instruct"
DEFAULT_EMBEDDING_QUERY_INSTRUCTION = (
    "Given a financial question, retrieve relevant passages that answer the question"
)
EMBEDDING_MODEL_CHOICES = {
    "e5-mistral": DEFAULT_EMBEDDING_MODEL,
    "finlang": "FinLang/finance-embeddings-investopedia",
    "bge-m3": "BAAI/bge-m3",
    "all-mpnet": "sentence-transformers/all-mpnet-base-v2",
}

# Instruction-based (decoder-only) embedding models. Queries are prefixed with
# "Instruct: <task>\nQuery: " before encoding; documents are embedded as-is.
INSTRUCT_EMBEDDING_SPECS: dict[str, dict] = {
    DEFAULT_EMBEDDING_MODEL: {
        "query_instruction": DEFAULT_EMBEDDING_QUERY_INSTRUCTION,
        "max_seq_length": 4096,
        "batch_size": 4,
    },
    # Reserved: same usage as e5-mistral once Fin-E5 weights become available.
    "FinanceMTEB/FinE5": {
        "query_instruction": DEFAULT_EMBEDDING_QUERY_INSTRUCTION,
        "max_seq_length": 4096,
        "batch_size": 4,
    },
}

# Default reranker: LLM-based cross-encoder (gemma-2b), scored with
# FlagEmbedding.FlagLLMReranker instead of sentence-transformers CrossEncoder.
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-gemma"
RERANKER_MODEL_CHOICES = {
    "bge-gemma": DEFAULT_RERANKER_MODEL,
    "bge-m3": "BAAI/bge-reranker-v2-m3",
    "mini": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "electra": "cross-encoder/ms-marco-electra-base",
}

# Decoder-only rerankers that need FlagLLMReranker (yes-token logit scoring).
LLM_RERANKER_MODELS = {
    "BAAI/bge-reranker-v2-gemma",
    "BAAI/bge-reranker-v2-minicpm-layerwise",
    "BAAI/bge-reranker-v2.5-gemma2-lightweight",
}
RERANKER_MAX_LENGTH = 512
RERANK_PREDICT_BATCH_SIZE = int(os.getenv("FINAGENT_RERANK_BATCH_SIZE", "8"))

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
