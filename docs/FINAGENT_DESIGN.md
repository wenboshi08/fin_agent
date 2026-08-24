# FinAgent — System Design Document

## 1. Overview

FinAgent is a **benchmark-driven financial RAG system**. The current implementation
is an MVP focused on the [ICAIF-24 Finance RAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge/overview)
and its seven public datasets.

The system is designed to answer two questions in a reproducible and low-cost way:

1. **Retrieval quality**: Can hybrid retrieval find the right passages from financial documents?
2. **Generation quality**: Can an LLM produce grounded, citation-backed answers from those passages?

The project deliberately uses the **LangChain ecosystem** as its foundation so that
the pipeline stays close to the reference implementation
([shivam1423/Financial-RAG-System](https://github.com/shivam1423/Financial-RAG-System)),
is easy to extend, and integrates cleanly with RAGAS evaluation.

### 1.1 Scope

**In scope (MVP)**

- Seven ICAIF-24 benchmark datasets.
- Dataset-aware chunking with multiple configurable strategies.
- Dense + BM25 hybrid retrieval, RRF fusion, MMR diversity, and LLM reranking.
- Grounded answer generation using DeepSeek or Qwen through OpenAI-compatible APIs.
- NDCG@10 retrieval evaluation and RAGAS answer evaluation.
- Experiment configuration through `configs/*.json`.
- Local CLI, synthetic demo, and Google Colab notebook.

**Out of scope (future phases)**

- Chinese research reports / financial filings parsing and OCR.
- Structured financial fact extraction and formula engines.
- Heading-tree based structural routing.
- Production service APIs, authentication, and multi-tenancy.
- Vector databases beyond local ChromaDB (Milvus, Elasticsearch, etc.).

---

## 2. System Requirements

### 2.1 Datasets

| Dataset | Docs | Queries | Type | Domain |
|---|---:|---:|---|---|
| FinanceBench | 180 | 150 | Passage | 10-K annual reports |
| FinQABench | 92 | 100 | Passage | 10-K hallucination-aware |
| FinDER | 13,867 | 216 | Passage | 10-K domain jargon |
| TATQA | 2,756 | 1,663 | Tabular | Hybrid table + text |
| FinQA | 2,789 | 1,147 | Tabular | Multi-step numerical reasoning |
| ConvFinQA | 2,066 | 421 | Tabular | Multi-turn conversational |
| MultiHiertt | 10,475 | 974 | Tabular | Multi-hop hierarchical tables |

### 2.2 Functional Requirements

- Load `corpus.jsonl`, `queries.jsonl`, and `qrels.tsv` for any dataset.
- Chunk documents using dataset-aware, fixed, semantic, or TF-IDF-style strategies.
- Build and cache a persistent vector index and a BM25 index.
- Retrieve top-k document IDs per query through a hybrid pipeline.
- Generate grounded answers with citations when an LLM is configured.
- Evaluate retrieval with NDCG@10.
- Evaluate generation with RAGAS (faithfulness, answer relevancy, context utilization).
- Record experiment results for regression comparison.

### 2.3 Non-Functional Requirements

| Requirement | Design Choice |
|---|---|
| Reproducibility | `uv.lock`, pinned dependency ranges, fixed dataset paths, deterministic sampling seeds |
| Low cost | Retrieval benchmark requires no LLM; generation uses low-cost DeepSeek/Qwen APIs; MultiQuery is optional |
| Portability | Runs on macOS x86_64, Linux, and Google Colab with the same `uv sync` flow |
| GPU compute | Default LLM embedding + reranker load sequentially (embed → offload to CPU → rerank), peak ~14 GB VRAM (Colab L4); CPU-friendly legacy models selectable via aliases |
| Extensibility | LangChain abstractions allow swapping embedding models, rerankers, and LLM providers |
| Testability | Pure functions for chunking/NDCG; synthetic demo requires no external services |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User / CLI / Colab                         │
│         main.py · eval.py · scripts/summarize_results.py           │
└───────────────────────────────┬─────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Configuration Layer                        │
│         configs/*.json · finrag/config.py · .env                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                ▼
┌───────────────────────────────┴─────────────────────────────────────┐
│                         Pipeline Layer                             │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐  │
│  │  Chunking  │ → │  Indexing  │ → │ Retrieval  │ → │ Generation │  │
│  │ LangChain  │   │ Chroma+BM25│   │ Hybrid+RRF │   │ DeepSeek/  │  │
│  │ splitters  │   │            │   │ +MMR+Rerank│   │ Qwen       │  │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Evaluation Layer                           │
│         NDCG@10 · RAGAS · regression CSV / charts                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1 Data Flow

1. **Data loading** (`data.py`)
   - Reads `corpus.jsonl`, `queries.jsonl`, and `qrels.tsv`.
   - Builds a corpus lookup `{corpus_id: document}`.

2. **Chunking** (`chunking.py`)
   - Uses `RecursiveCharacterTextSplitter` for dataset-aware and fixed strategies.
   - Uses custom sentence/paragraph grouping for semantic and TF-IDF-style strategies.
   - Produces `ChunkResult` with text, IDs, original corpus IDs, and LangChain `Document` objects.

3. **Indexing** (`vectorstore.py`, `retrieval.py`)
   - Embeds chunks with `intfloat/e5-mistral-7b-instruct` (documents as-is; queries get a task instruction prefix).
   - Persists vectors in `langchain_chroma.Chroma`.
   - Builds a `BM25Retriever` from the same chunk documents.

4. **Retrieval** (`retrieval.py`)
   - Optional MultiQuery expansion via LLM.
   - Dense retrieval through Chroma (MMR for passage, similarity for tabular).
   - Sparse retrieval through BM25.
   - RRF fusion of dense and sparse rankings.
   - Best-chunk selection via BM25 scores.
   - LLM reranking (`BAAI/bge-reranker-v2-gemma` via `FlagLLMReranker`) to produce the final top-k.

5. **Generation** (`generation.py`)
   - Builds a `ChatPromptTemplate` with the retrieved excerpts.
   - Calls DeepSeek/Qwen via `ChatOpenAI`.
   - Parses inline `[doc_id]` citations and returns source metadata.

6. **Evaluation** (`evaluation.py`, `ragas_eval.py`)
   - NDCG@10 for retrieval.
   - RAGAS metrics for generated answers.
   - Results appended to `results/regression.csv`.

---

## 4. Component Design

### 4.1 Configuration (`config.py`)

- Centralizes dataset paths, chunk sizes, fetch-k, rerank-top-n, and model defaults.
- Dataset-specific values are aligned with the upstream reference.

### 4.2 Chunking (`chunking.py`)

Strategies:

| Strategy | Behavior |
|---|---|
| `dataset-aware` | Default. Passage uses sentence/paragraph separators; tabular uses pipe `|` to preserve table rows |
| `fixed` | Fixed-size character chunks with overlap |
| `semantic` | Sentence-level grouping up to a target size |
| `tfidf` | Paragraph-boundary grouping as a lightweight topic-aware approximation |

Uses LangChain `Document` objects so downstream BM25 and Chroma components accept the same representation.

### 4.3 Models (`models.py`)

- `E5InstructEmbeddings` (a `SentenceTransformer` wrapper that applies a task
  instruction prefix to queries) for instruction-based LLM embeddings
  (`intfloat/e5-mistral-7b-instruct`, and `FinanceMTEB/FinE5` once its weights
  become accessible); `HuggingFaceEmbeddings` for encoder-based models.
- `FlagLLMReranker` (FlagEmbedding) for decoder-only rerankers
  (`BAAI/bge-reranker-v2-gemma`); `sentence_transformers.CrossEncoder` for
  classic cross-encoders.
- `ChatOpenAI` for DeepSeek and Qwen (OpenAI-compatible endpoints).
- A rate-limited `get_eval_llm` for RAGAS judge calls.

### 4.4 Vector Store (`vectorstore.py`)

- Wraps `langchain_chroma.Chroma`.
- Supports `force_rebuild`, persistent disk cache, batched `add_texts`, and MMR via Chroma.

### 4.5 Retrieval (`retrieval.py`)

```
query
  ├─ [MultiQuery] → multiple query variants
  ├─ Dense: Chroma similarity / MMR
  ├─ BM25: BM25Retriever
  ├─ RRF fusion
  ├─ Best chunk per candidate
  └─ LLM rerank (bge-reranker-v2-gemma) → top-k
```

### 4.6 Generation (`generation.py`)

- System prompt enforces grounded answers and citation syntax.
- Only retrieved excerpts are provided as context.
- Citations are extracted from the answer and mapped back to source documents.

### 4.7 Pipeline (`pipeline.py`)

- `prepare_retriever`: loads data, chunks, builds/loads vector store and BM25 index.
- `run_dataset_benchmark`: iterates all queries, computes NDCG@10.
- `run_rag_query`: retrieves and generates a single grounded answer.

### 4.8 Evaluation

- `evaluation.py`: pure NDCG@k implementation.
- `ragas_eval.py`: RAGAS harness with `compat.py` and `RunConfig(timeout=600, max_retries=5, max_wait=120)`.
- `regression.py`: CSV tracking and comparison charts.

### 4.9 CLI

- `main.py`: retrieval benchmark / single-query RAG.
- `eval.py`: RAGAS evaluation with `configs/*.json` support and CLI overrides.
- `scripts/summarize_results.py`: prints the NDCG summary table.
- `scripts/download_data.sh`: Kaggle download helper.
- `scripts/export_colab_zip.sh`: Colab packaging.
- `scripts/demo_synthetic.py`: dependency-free demo.

---

## 5. LangChain Adoption

### 5.1 Why LangChain?

The decision to use LangChain is a **system design choice**, not an implementation
detail. It affects maintainability, comparability, and future extensibility.

1. **Pre-built, battle-tested components**
   - Chunking, vector stores, retrievers, embeddings, and chat models are provided
     by LangChain and its official partner packages.
   - This reduces custom code and the risk of subtle bugs in document splitting,
     vector-store persistence, and retriever semantics.

2. **RAGAS integration**
   - RAGAS is the standard reference-free evaluation framework for RAG.
   - It natively wraps LangChain LLMs and embeddings, so LangChain is the least
     friction path to high-quality answer evaluation.

3. **Alignment with the reference project**
   - The upstream `shivam1423/Financial-RAG-System` uses the same LangChain stack.
   - Using the same abstractions makes our NDCG@10 and RAGAS results directly
     comparable with the published baseline.

4. **Model and retriever portability**
   - Embedding models, rerankers, and LLM providers can be swapped through
     configuration without rewriting pipeline code.
   - DeepSeek and Qwen both expose OpenAI-compatible APIs and are consumed through
     `ChatOpenAI`.

5. **Future-proofing**
   - Later phases (agents, structured output, tracing, heading-tree routing) can
     reuse LangChain/LangGraph primitives instead of introducing a second framework.

### 5.2 Where LangChain Is Used

| Area | LangChain Component | Effect |
|---|---|---|
| Chunking | `RecursiveCharacterTextSplitter`, `Document` | Robust splitting and uniform chunk metadata |
| Embeddings | `HuggingFaceEmbeddings`, `E5InstructEmbeddings` | Pluggable embedding models (e5-mistral/Fin-E5 family, FinLang, BGE, MPNet) |
| Vector store | `langchain_chroma.Chroma` | Persistent caching, similarity search, MMR |
| Sparse retrieval | `BM25Retriever` | Direct integration with LangChain documents |
| LLM | `ChatOpenAI` | DeepSeek/Qwen via OpenAI-compatible endpoints |
| Prompts | `ChatPromptTemplate` | Grounded generation with citations |
| Evaluation | RAGAS + LangChain wrappers | Faithfulness, relevancy, context utilization |

### 5.3 Effects Achieved

- **Reduced custom glue**: the pipeline is mostly declarative composition of
  LangChain components.
- **Direct benchmark comparability**: the same component stack as the reference repo.
- **Working RAGAS evaluation**: `compat.py` patches the removed
  `langchain_community.chat_models.vertexai` module; `RunConfig` makes evaluation
  robust under API rate limits.
- **Multi-provider LLM support**: DeepSeek and Qwen are selected by configuration.
- **Reproducible environment**: `uv` + pinned constraints
  (`torch<2.3`, `onnxruntime<1.20`, `transformers<5`, `numpy<2`) solve macOS x86_64
  wheel availability and keep the stack compatible.

### 5.4 Trade-offs

- **Dependency weight**: LangChain and its integrations add many transitive packages.
- **Version coupling**: LangChain, Transformers, and PyTorch must be pinned together
  to avoid platform/API incompatibilities.
- **Abstraction overhead**: debugging sometimes requires tracing through LangChain
  wrappers.
- **Breaking changes**: LangChain evolves quickly; the compatibility patch and
  version pins mitigate this for the current MVP.

---

## 6. Reliability and Cost-Effectiveness

### 6.1 Reliability

- **Reproducible builds**: `uv.lock` and explicit version ranges.
- **Persistent caching**: ChromaDB stores embeddings on disk; unchanged datasets
  skip re-embedding.
- **Graceful degradation**: if no LLM API key is present, retrieval benchmarks and
  the synthetic demo still run.
- **Evaluation robustness**: RAGAS uses `RunConfig` with generous timeouts and
  retries; `compat.py` prevents import-time failures.
- **Data validation**: unit tests cover chunking and NDCG behavior.

### 6.2 Cost-Effectiveness

- **Retrieval-only benchmarks require no LLM calls**.
- **Cheap generation**: DeepSeek/Qwen APIs are used only when generating answers
  or running RAGAS.
- **Optional MultiQuery**: disabled by default in summary scripts and can be toggled
  for ablation.
- **Small contexts**: only top-k retrieved excerpts are sent to the LLM.
- **Local models for evaluation**: embedding and reranker models run locally;
  RAGAS uses the same low-cost API providers.

---

## 7. Evaluation and Expected Results

### 7.1 Retrieval (NDCG@10)

| Dataset | Reference NDCG@10 target |
|---|---|
| FinanceBench | ≥ 0.82 |
| FinQABench | ≥ 0.94 |
| FinDER | ≥ 0.71 |
| TATQA | ≥ 0.78 |
| FinQA | ≥ 0.78 |
| ConvFinQA | ≥ 0.75 |
| MultiHiertt | ≥ 0.77 |

### 7.2 Generation (RAGAS)

Target metrics:

| Metric | Target |
|---|---|
| Faithfulness | ≥ 0.90 |
| Answer Relevancy | ≥ 0.90 |
| Context Utilization | ≥ 0.75 |

### 7.3 How to Produce the Summary

```bash
# Run all datasets and print a Markdown summary
uv run python scripts/summarize_results.py

# Or reuse saved results
uv run python main.py --all --save-results results/all.json
uv run python scripts/summarize_results.py --from-json results/all.json
```

---

## 8. Project Structure

```
finagent/
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py
├── eval.py
├── configs/
├── finrag/
│   ├── config.py
│   ├── data.py
│   ├── chunking.py
│   ├── models.py
│   ├── vectorstore.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── pipeline.py
│   ├── evaluation.py
│   ├── ragas_eval.py
│   ├── regression.py
│   └── compat.py
├── scripts/
│   ├── download_data.sh
│   ├── export_colab_zip.sh
│   ├── summarize_results.py
│   └── demo_synthetic.py
├── colab/
│   └── FinAgent_Colab.ipynb
├── tests/
└── docs/
    ├── FINAGENT_DESIGN.md
    ├── MVP_SCOPE.md
    └── observability_slo.md
```

---

## 9. Future Work

- Heading-tree based structural routing (inspired by AsyncBuilds/FinRag).
- Table-aware chunking and structured table representation.
- Structured financial fact extraction and formula engines for numerical reasoning.
- Full conversational state for ConvFinQA.
- Production FastAPI/Streamlit service and SLO monitoring.
- Agentic orchestration with LangGraph for multi-hop questions.
