# FinAgent MVP

FinAgent is a **benchmark-driven financial RAG system** built around the
[ICAIF-24 Finance RAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge/overview).
It provides a reproducible, low-cost pipeline for hybrid retrieval and grounded
answer generation over **7 public financial datasets**.

The reference implementation is
[shivam1423/Financial-RAG-System](https://github.com/shivam1423/Financial-RAG-System).

---

## Highlights

- **7 ICAIF-24 datasets**: FinanceBench, FinQABench, FinDER, TATQA, FinQA, ConvFinQA, MultiHiertt.
- **Hybrid retrieval**: Dense (ChromaDB + FinLang embeddings) + BM25 → RRF fusion → MMR diversity → Cross-Encoder rerank.
- **Grounded generation**: DeepSeek / Qwen via OpenAI-compatible API, with source citations.
- **LangChain-based architecture**: built on `langchain-chroma`, `langchain-community`, `langchain-huggingface`, `langchain-openai`, and `langchain-text-splitters`.
- **Two-layer evaluation**: NDCG@10 for retrieval quality, RAGAS for answer faithfulness/relevancy/context utilization.
- **Configurable experiments**: all RAGAS experiment settings come from `configs/*.json`, with CLI overrides.
- **Runs anywhere**: local CLI, dependency-free synthetic demo, and a Google Colab notebook.

---

## System Design Overview

```
                    ┌──────────────────────────────────────────┐
                    │              User / CLI / Colab          │
                    └────────────────────┬─────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │                Orchestration             │
                    │  main.py · eval.py · configs/*.json      │
                    └────────────────────┬─────────────────────┘
                                         ▼
   ┌─────────────────────────┬──────────────────────────┬──────────────────────┐
   │      Chunking           │        Indexing          │      Retrieval        │
   │ LangChain text splitters│ ChromaDB + BM25          │ Dense + BM25 + RRF    │
   │ dataset-aware / fixed / │ FinLang embeddings       │ + MMR + reranker      │
   │ semantic / tfidf        │ persistent cache         │                      │
   └─────────────────────────┴──────────────────────────┴──────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │              Generation                  │
                    │ ChatPromptTemplate + ChatOpenAI          │
                    │ DeepSeek / Qwen, grounded citations      │
                    └────────────────────┬─────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │              Evaluation                  │
                    │ NDCG@10 · RAGAS · regression CSV         │
                    └──────────────────────────────────────────┘
```

### Data flow

1. Load `corpus.jsonl`, `queries.jsonl`, and `qrels.tsv` from `Dataset/`.
2. Split each document into chunks using a configurable strategy.
3. Embed chunks with `FinLang/finance-embeddings-investopedia` and store them in a persistent ChromaDB collection.
4. Build a BM25 index over the same chunks.
5. For each query:
   - optional MultiQuery expansion,
   - dense retrieval and BM25 retrieval,
   - RRF fusion,
   - MMR diversity selection (passage datasets),
   - Cross-Encoder reranking,
   - grounded answer generation with citations.
6. Evaluate retrieval with NDCG@10 and generation with RAGAS.

---

## Why LangChain?

FinAgent intentionally uses the **LangChain ecosystem** instead of a fully custom
implementation. The main reasons are:

1. **Standard building blocks**
   - `RecursiveCharacterTextSplitter` for robust chunking.
   - `Chroma` from `langchain-chroma` for persistent vector storage and built-in MMR.
   - `BM25Retriever` from `langchain-community` for sparse retrieval.
   - `HuggingFaceEmbeddings` for pluggable embedding models.
   - `ChatOpenAI` for DeepSeek/Qwen through OpenAI-compatible endpoints.
2. **RAGAS compatibility**
   - RAGAS evaluation natively wraps LangChain LLMs and embeddings. Using LangChain avoids a separate integration layer.
3. **Alignment with the reference project**
   - The upstream `shivam1423/Financial-RAG-System` is LangChain-based. Reusing the same abstractions makes results directly comparable.
4. **Maintainability**
   - Less custom glue code, clearer component boundaries, and easier model/retriever swapping for ablation experiments.
5. **Ecosystem velocity**
   - LangChain provides active maintenance, community patterns, and a broad set of integrations useful for future phases (agents, structured output, tracing).

### Where LangChain is used

| Module | LangChain component | Purpose |
|---|---|---|
| `chunking.py` | `langchain_text_splitters.RecursiveCharacterTextSplitter`, `langchain_core.documents.Document` | Document splitting and chunk representation |
| `models.py` | `langchain_huggingface.HuggingFaceEmbeddings`, `langchain_openai.ChatOpenAI` | Embedding and DeepSeek/Qwen LLM clients |
| `vectorstore.py` | `langchain_chroma.Chroma` | Persistent vector store, similarity search, MMR |
| `retrieval.py` | `langchain_community.retrievers.BM25Retriever` | Sparse retrieval |
| `generation.py` | `langchain_core.prompts.ChatPromptTemplate` | Prompt construction and grounded generation |
| `ragas_eval.py` | RAGAS + LangChain wrappers | Faithfulness, answer relevancy, context utilization |

### Effects achieved with LangChain

- **Faster iteration**: retrieval, reranking, and generation share the same document and model abstractions.
- **Direct comparability**: the pipeline maps 1:1 to the upstream LangChain reference.
- **RAGAS works out of the box**: a small `compat.py` patch handles the
  `langchain_community.chat_models.vertexai` import issue, and `RunConfig`
  controls timeouts/retries during evaluation.
- **Pluggable models**: embedding and reranker choices are configurable through
  CLI flags and JSON configs.
- **Reproducible environment**: `uv` lockfile plus explicit version constraints
  (`torch<2.3`, `onnxruntime<1.20`, `transformers<5`, `numpy<2`) make the project
  installable on macOS x86_64 and in Colab.

---

## Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Download the Kaggle data
#    Place your Kaggle API token at ~/.kaggle/access_token (or ~/.kaggle/kaggle.json)
bash scripts/download_data.sh

# 3. Configure DeepSeek / Qwen
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY or QWEN_API_KEY

# 4. Run a retrieval benchmark
uv run python main.py --dataset financebench

# 5. Run all datasets
uv run python main.py --all

# 6. Ask a single question
uv run python main.py --dataset financebench \
  --query "What was Apple's revenue in FY2022?"

# 7. Run RAGAS evaluation (requires an LLM API key)
uv run python eval.py --config deepseek_baseline --dataset financebench
```

---

## Commands

### Retrieval benchmark

```bash
# Single dataset
uv run python main.py --dataset financebench

# All datasets
uv run python main.py --all

# Save results as JSON
uv run python main.py --all --save-results results/all.json
```

### Single-query RAG

```bash
uv run python main.py --dataset financebench \
  --query "What was Acme's total revenue in FY2024?" \
  --provider deepseek --model deepseek-chat
```

### RAGAS evaluation

```bash
# Load experiment parameters from configs/*.json
uv run python eval.py --config deepseek_baseline --dataset financebench
uv run python eval.py --config qwen_baseline --dataset tatqa --n 20
```

### Results summary

```bash
# Run all datasets and print a Markdown summary table
uv run python scripts/summarize_results.py

# Reuse previously saved results
uv run python scripts/summarize_results.py --from-json results/all.json
```

### Synthetic demo (no dependencies, no API)

```bash
uv run python scripts/demo_synthetic.py
```

---

## Google Colab

```bash
# Package the project (without Dataset)
bash scripts/export_colab_zip.sh

# Or include Dataset (~68 MB)
bash scripts/export_colab_zip.sh --with-data
```

Then open [`colab/FinAgent_Colab.ipynb`](colab/FinAgent_Colab.ipynb), upload
`finagent_colab.zip`, and run the cells. The notebook can automatically download
the Kaggle data when an access token is provided.

---

## Project Structure

```
finagent/
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py                  # Retrieval benchmark + single-query RAG CLI
├── eval.py                  # RAGAS evaluation CLI
├── configs/                 # Experiment JSON configs
├── finrag/
│   ├── config.py            # Dataset paths, model defaults, dataset configs
│   ├── data.py              # JSONL / qrels loaders
│   ├── chunking.py          # LangChain-based chunking strategies
│   ├── models.py            # Embedding, reranker, DeepSeek/Qwen LLM
│   ├── vectorstore.py       # langchain_chroma.Chroma wrapper
│   ├── retrieval.py         # Dense + BM25 + RRF + MMR + rerank
│   ├── generation.py        # Grounded generation with citations
│   ├── pipeline.py          # High-level orchestration
│   ├── evaluation.py        # NDCG@k
│   ├── ragas_eval.py        # RAGAS harness
│   ├── regression.py        # Experiment regression tracking
│   └── compat.py            # RAGAS / LangChain compatibility patch
├── scripts/
│   ├── download_data.sh     # Kaggle download script
│   ├── export_colab_zip.sh  # Colab packaging script
│   ├── summarize_results.py # NDCG summary table generator
│   └── demo_synthetic.py    # Dependency-free local demo
├── colab/
│   └── FinAgent_Colab.ipynb # One-click Colab notebook
├── tests/                   # Unit tests
└── docs/                    # Design and scope documents
```

---

## Documentation

- `docs/FINAGENT_DESIGN.md` — System design document
- `docs/MVP_SCOPE.md` — Narrowed MVP scope
- `docs/observability_slo.md` — SLO / monitoring design
