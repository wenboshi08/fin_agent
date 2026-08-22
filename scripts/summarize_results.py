#!/usr/bin/env python3
"""Print a Results Summary table like the Kaggle notebook.

By default it runs all 7 datasets and prints:

| Dataset | Type | Local NDCG@10 | Notes |
|---------|------|---------------|-------|

If you already have a saved retrieval results JSON (from `main.py --save-results`),
use `--from-json results.json` to avoid re-running retrieval.

Examples:
  python scripts/summarize_results.py
  python scripts/summarize_results.py --no-multiquery
  python scripts/summarize_results.py --from-json results/all.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrag.config import DATASET_CONFIGS

# Notes copied from the reference notebook's Results Summary.
NOTES = {
    "finder": "10-K jargon + abbreviations",
    "financebench": "Natural language financial Q&A",
    "finqabench": "Hallucination-aware retrieval",
    "finqa": "Multi-step numerical reasoning",
    "tatqa": "Hybrid table + text",
    "convfinqa": "Multi-turn conversational",
    "multiheirtt": "Multi-hop hierarchical tables",
}

TYPE_LABEL = {
    "passage": "Passage",
    "tabular": "Tabular",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print FinAgent Results Summary table")
    parser.add_argument("--from-json", type=Path, default=None,
                        help="Load retrieval results from a saved JSON instead of running.")
    parser.add_argument("--no-multiquery", action="store_true", default=False)
    parser.add_argument("--provider", choices=["deepseek", "qwen"], default="deepseek")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_json is not None:
        data = json.loads(Path(args.from_json).read_text())
        ndcg_map = {}
        # If the JSON is {dataset: {query_id: {doc_id: score}}}, compute NDCG.
        from finrag.data import load_qrels
        from finrag.evaluation import compute_ndcg

        for ds in DATASET_CONFIGS:
            if ds not in data:
                continue
            qrels_path = ROOT / "Dataset" / DATASET_CONFIGS[ds].qrels_file
            qrels = load_qrels(qrels_path)
            ndcg_map[ds] = compute_ndcg(qrels, data[ds], k=args.top_k)
    else:
        from finrag.pipeline import run_dataset_benchmark

        ndcg_map = {}
        for ds in DATASET_CONFIGS:
            print(f"Running {ds} ...", file=sys.stderr)
            result = run_dataset_benchmark(
                ds,
                top_k=args.top_k,
                use_multiquery=not args.no_multiquery,
                provider=args.provider,
                model=args.model,
                mmr_lambda=args.mmr_lambda,
            )
            ndcg_map[ds] = result.ndcg

    print("\n## 📈 Results Summary\n")
    print("| Dataset | Type | Local NDCG@10 | Notes |")
    print("|---------|------|---------------|-------|")
    for ds, cfg in DATASET_CONFIGS.items():
        ndcg = ndcg_map.get(ds, float("nan"))
        ndcg_str = f"{ndcg:.4f}" if ndcg == ndcg else "N/A"
        print(f"| {ds} | {TYPE_LABEL.get(cfg.dataset_type, cfg.dataset_type)} | {ndcg_str} | {NOTES.get(ds, '')} |")

    print()


if __name__ == "__main__":
    main()
