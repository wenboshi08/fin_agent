"""FinAgent MVP CLI — retrieval benchmark and single-query RAG.

Examples:
  python main.py --dataset financebench
  python main.py --all
  python main.py --dataset tatqa --query "What was total revenue in FY2024?"
  python main.py --dataset finqa --no-multiquery --provider qwen --model qwen-plus
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from finrag import config
from finrag.config import DATASET_CONFIGS

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="finagent",
        description="ICAIF-24 Financial RAG — hybrid retrieval + grounded generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", choices=list(DATASET_CONFIGS), metavar="NAME",
                        help="Dataset to use.")
    target.add_argument("--all", action="store_true",
                        help="Run retrieval benchmark over all datasets sequentially.")

    parser.add_argument("--query", "-q", type=str, metavar="QUESTION",
                        help="Ask a financial question (enables full RAG).")

    parser.add_argument("--top-k", type=int, default=None, metavar="K",
                        help="Documents to retrieve/return (benchmark default 10, RAG default 5).")
    parser.add_argument("--rebuild", action="store_true", default=False,
                        help="Re-embed corpus even if ChromaDB cache exists.")
    parser.add_argument("--no-multiquery", action="store_true", default=False,
                        help="Disable MultiQuery LLM expansion.")
    parser.add_argument("--provider", choices=["deepseek", "qwen"], default=None,
                        help="LLM provider (default: FINAGENT_LLM_PROVIDER or deepseek).")
    parser.add_argument("--model", type=str, default=None, metavar="MODEL",
                        help="Model override (e.g. deepseek-chat, qwen-plus).")
    parser.add_argument("--chunk-strategy", choices=["dataset-aware", "fixed", "semantic", "tfidf"],
                        default="dataset-aware", help="Chunking strategy.")
    parser.add_argument("--embedding-model", choices=list(config.EMBEDDING_MODEL_CHOICES),
                        default="e5-mistral", help="Embedding model alias.")
    parser.add_argument("--reranker", choices=list(config.RERANKER_MODEL_CHOICES),
                        default="bge-gemma", help="Reranker model alias.")
    parser.add_argument("--mmr-lambda", type=float, default=config.DEFAULT_MMR_LAMBDA,
                        help="MMR lambda (0 disables MMR).")
    parser.add_argument("--no-mmr", action="store_true", default=False,
                        help="Disable MMR diversity step.")
    parser.add_argument("--save-results", type=Path, metavar="PATH",
                        help="Save retrieval results as JSON.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
                        help="Logging verbosity.")
    return parser.parse_args()


def _save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved -> %s", path)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for lib in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    provider = args.provider or config.DEFAULT_LLM_PROVIDER
    mmr_lambda = 0.0 if args.no_mmr else args.mmr_lambda

    from finrag.pipeline import run_dataset_benchmark, run_rag_query

    if args.query:
        if args.all:
            print("--query cannot be combined with --all. Pick one dataset.")
            sys.exit(1)
        top_k = args.top_k if args.top_k is not None else config.DEFAULT_RAG_TOP_K
        result = run_rag_query(
            args.query,
            args.dataset,
            top_k=top_k,
            use_multiquery=not args.no_multiquery,
            force_rebuild=args.rebuild,
            provider=provider,
            model=args.model,
            chunk_strategy=args.chunk_strategy,
            embedding_model=config.EMBEDDING_MODEL_CHOICES[args.embedding_model],
            reranker_model=config.RERANKER_MODEL_CHOICES[args.reranker],
            mmr_lambda=mmr_lambda,
        )
        print(result.pretty())
        return

    if args.all:
        top_k = args.top_k or config.DEFAULT_TOP_K
        all_results = {}
        print("\nRunning retrieval benchmark over all datasets...\n")
        for ds in DATASET_CONFIGS:
            r = run_dataset_benchmark(
                ds,
                top_k=top_k,
                use_multiquery=not args.no_multiquery,
                force_rebuild=args.rebuild,
                provider=provider,
                model=args.model,
                chunk_strategy=args.chunk_strategy,
                embedding_model=config.EMBEDDING_MODEL_CHOICES[args.embedding_model],
                reranker_model=config.RERANKER_MODEL_CHOICES[args.reranker],
                mmr_lambda=mmr_lambda,
            )
            ndcg = f"{r.ndcg:.4f}" if r.ndcg == r.ndcg else "N/A"
            print(f" {ds:<14} NDCG@10={ndcg} queries={r.num_queries} chunks={r.num_chunks} "
                  f"time={r.elapsed_sec:.1f}s errors={len(r.errors)}")
            all_results[ds] = r.results
        if args.save_results:
            _save_json(all_results, args.save_results)
        return

    top_k = args.top_k or config.DEFAULT_TOP_K
    r = run_dataset_benchmark(
        args.dataset,
        top_k=top_k,
        use_multiquery=not args.no_multiquery,
        force_rebuild=args.rebuild,
        provider=provider,
        model=args.model,
        chunk_strategy=args.chunk_strategy,
        embedding_model=config.EMBEDDING_MODEL_CHOICES[args.embedding_model],
        reranker_model=config.RERANKER_MODEL_CHOICES[args.reranker],
        mmr_lambda=mmr_lambda,
    )
    ndcg = f"{r.ndcg:.4f}" if r.ndcg == r.ndcg else "N/A"
    print(f"\n{'=' * 44}")
    print(f" Dataset : {r.name}")
    print(f" NDCG@10 : {ndcg}")
    print(f" Queries : {r.num_queries}")
    print(f" Chunks  : {r.num_chunks}")
    print(f" Time    : {r.elapsed_sec:.1f}s")
    if r.errors:
        print(f" Errors  : {len(r.errors)}")
    print(f"{'=' * 44}\n")

    if args.save_results:
        _save_json(r.results, args.save_results)


if __name__ == "__main__":
    main()
