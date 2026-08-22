"""RAGAS evaluation CLI — configs come from configs/*.json, CLI flags override.

Examples:
  python eval.py --config deepseek_baseline --dataset financebench
  python eval.py --config configs/qwen_baseline.json --dataset tatqa
  python eval.py --config multiquery_k5 --dataset finqa --n 20
  python eval.py --compare --output results/ragas_comparison.png
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Must be imported before ragas: patches langchain_community.chat_models.vertexai
from finrag import compat as _compat  # noqa: F401
from finrag import config
from finrag.config import DATASET_CONFIGS, EMBEDDING_MODEL_CHOICES, RERANKER_MODEL_CHOICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for FinAgent MVP")
    parser.add_argument("--config", default="baseline_k5",
                        help="Experiment config name (configs/<name>.json) or explicit .json path.")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS), default=None,
                        help="Dataset name. Optional if config file has 'dataset'.")
    parser.add_argument("--provider", choices=["deepseek", "qwen"], default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n", type=int, default=None, help="Number of queries to evaluate.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-multiquery", action="store_true", default=None,
                        help="Disable MultiQuery (overrides config 'multiquery').")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--chunk-strategy", choices=["dataset-aware", "fixed", "semantic", "tfidf"],
                        default=None)
    parser.add_argument("--embedding-model", choices=list(EMBEDDING_MODEL_CHOICES), default=None)
    parser.add_argument("--reranker", choices=list(RERANKER_MODEL_CHOICES), default=None)
    parser.add_argument("--mmr-lambda", type=float, default=None)
    parser.add_argument("--no-mmr", action="store_true", default=None,
                        help="Disable MMR (overrides config 'mmr_lambda').")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Compare existing regression CSV results and save a chart.")
    parser.add_argument("--output", type=str, default=None,
                        help="Chart output path for --compare.")
    return parser.parse_args()


def load_config_file(args) -> tuple[dict, str]:
    """Load config from configs/<name>.json. Returns (config_dict, config_label)."""
    label = args.config or "baseline_k5"
    path = Path(label)
    if not path.suffix:
        candidate = config.CONFIGS_DIR / f"{label}.json"
        if candidate.exists():
            path = candidate
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        label = path.stem if path.suffix == ".json" else label
        logging.getLogger(__name__).info("Loaded config from %s", path)
        return data, label
    logging.getLogger(__name__).warning("Config file not found for %r — using defaults.", args.config)
    return {}, label


def resolve_settings(args, cfg: dict) -> dict:
    dataset = args.dataset or cfg.get("dataset")
    if not dataset:
        raise SystemExit("Missing --dataset (or 'dataset' in the config JSON).")

    provider = args.provider or cfg.get("provider") or config.DEFAULT_LLM_PROVIDER
    model = args.model or cfg.get("model")
    n = args.n if args.n is not None else cfg.get("n", 20)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)

    if args.no_multiquery is None:
        use_multiquery = bool(cfg.get("multiquery", True))
    else:
        use_multiquery = not args.no_multiquery

    top_k = args.top_k if args.top_k is not None else cfg.get("top_k", 5)
    chunk_strategy = args.chunk_strategy or cfg.get("chunk_strategy", "dataset-aware")
    embedding_alias = args.embedding_model or cfg.get("embedding_model", "finlang")
    reranker_alias = args.reranker or cfg.get("reranker", "bge")

    if args.no_mmr:
        mmr_lambda = 0.0
    elif args.mmr_lambda is not None:
        mmr_lambda = args.mmr_lambda
    else:
        mmr_lambda = float(cfg.get("mmr_lambda", config.DEFAULT_MMR_LAMBDA))

    return {
        "dataset": dataset,
        "provider": provider,
        "model": model,
        "n": n,
        "seed": seed,
        "use_multiquery": use_multiquery,
        "top_k": top_k,
        "chunk_strategy": chunk_strategy,
        "embedding_model": EMBEDDING_MODEL_CHOICES[embedding_alias],
        "reranker": RERANKER_MODEL_CHOICES[reranker_alias],
        "mmr_lambda": mmr_lambda,
    }


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

    if args.compare:
        from finrag.regression import compare_and_chart

        chart = compare_and_chart(datasets=list(DATASET_CONFIGS), metric="ragas_faithfulness", output=args.output)
        print(f"Comparison chart saved -> {chart}")
        return

    cfg, config_label = load_config_file(args)
    s = resolve_settings(args, cfg)

    # Lazy imports: avoid loading heavy model deps when only --compare is used
    from finrag.models import get_eval_llm, get_llm, get_reranker
    from finrag.pipeline import prepare_retriever
    from finrag.generation import generate_answer
    from finrag.retrieval import retrieve_and_rerank

    data = prepare_retriever(
        s["dataset"],
        chunk_strategy=s["chunk_strategy"],
        embedding_model=s["embedding_model"],
    )
    dataset_cfg = data["cfg"]
    judge = get_eval_llm(s["provider"], s["model"])
    if judge is None:
        print("RAGAS evaluation requires an LLM API key (DEEPSEEK_API_KEY or QWEN_API_KEY).")
        sys.exit(1)

    llm = get_llm(s["provider"], s["model"])
    reranker = get_reranker(s["reranker"])

    random.seed(s["seed"])
    sample = random.sample(data["queries"], min(s["n"], len(data["queries"])))

    retrieved_map: dict[str, list[tuple[str, float]]] = {}
    answers_map: dict[str, str] = {}

    for q in sample:
        qid = q["_id"]
        retrieved = retrieve_and_rerank(
            data["vectorstore"],
            data["bm25_retriever"],
            data["bm25_okapi"],
            data["chunks"],
            q["text"],
            reranker,
            dataset_type=dataset_cfg.dataset_type,
            fetch_k=dataset_cfg.fetch_k,
            rerank_top_n=dataset_cfg.rerank_top_n,
            k=s["top_k"],
            llm=llm if s["use_multiquery"] else None,
            mmr_lambda=s["mmr_lambda"],
        )
        retrieved_map[qid] = retrieved
        if llm is not None:
            result = generate_answer(
                q["text"], retrieved, data["corpus_lookup"], llm,
                top_k=s["top_k"],
                multiquery=s["use_multiquery"],
                model_name=getattr(llm, "model_name", f"{s['provider']}/{s['model']}"),
            )
            answers_map[qid] = result.answer
        else:
            answers_map[qid] = ""

    from finrag.ragas_eval import run_ragas_eval

    result = run_ragas_eval(
        data["queries"],
        retrieved_map,
        answers_map,
        data["corpus_lookup"],
        judge,
        dataset_name=s["dataset"],
        config_name=config_label,
        model_name=getattr(judge, "model_name", f"{s['provider']}/{s['model']}"),
        n_samples=s["n"],
        sample_seed=s["seed"],
    )
    print(result.summary())

    from finrag.regression import append_result

    append_result(dataset=s["dataset"], config_name=config_label, metric="ragas_faithfulness", value=result.faithfulness)
    append_result(dataset=s["dataset"], config_name=config_label, metric="ragas_answer_relevancy", value=result.answer_relevancy)
    append_result(dataset=s["dataset"], config_name=config_label, metric="ragas_context_utilization", value=result.context_utilization)


if __name__ == "__main__":
    main()
