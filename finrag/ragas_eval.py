"""RAGAS evaluation harness (faithfulness, answer relevancy, context utilization).

Uses LangChain ChatOpenAI wired to DeepSeek or Qwen OpenAI-compatible endpoints.
This module is only imported when running `eval.py`.
"""
from __future__ import annotations

# Must be imported before ragas: patches langchain_community.chat_models.vertexai
from . import compat as _compat  # noqa: F401

import logging
import random
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

_EVAL_EMBED_MODEL = "BAAI/bge-large-en-v1.5"


@dataclass
class RAGASResult:
    dataset: str
    config_name: str
    model: str
    n_samples: int
    faithfulness: float
    answer_relevancy: float
    context_utilization: float
    per_query_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def summary(self) -> str:
        return (
            f"{'=' * 52}\n"
            f" Dataset : {self.dataset}\n"
            f" Config : {self.config_name}\n"
            f" Model : {self.model}\n"
            f" Samples : {self.n_samples}\n"
            f"{'=' * 52}\n"
            f" Faithfulness : {self.faithfulness:.4f}\n"
            f" Answer Relevancy : {self.answer_relevancy:.4f}\n"
            f" Context Utilization : {self.context_utilization:.4f}\n"
            f"{'─' * 52}"
        )


def _build_metrics(llm):
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextUtilization, Faithfulness

    ragas_llm = LangchainLLMWrapper(llm)
    eval_emb = HuggingFaceEmbeddings(
        model_name=_EVAL_EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_emb = LangchainEmbeddingsWrapper(eval_emb)
    return (
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextUtilization(llm=ragas_llm),
    )


def _build_dataset(sample_queries, retrieved_map, answers_map, corpus_lookup, max_ctx_chars=1500):
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    samples = []
    query_ids = []
    for q in sample_queries:
        qid = q["_id"]
        if qid not in retrieved_map or qid not in answers_map:
            continue
        answer = answers_map.get(qid, "")
        if not answer or not answer.strip():
            continue
        contexts = []
        for doc_id, _ in retrieved_map[qid][:5]:
            doc = corpus_lookup.get(doc_id)
            if doc is None:
                continue
            text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            contexts.append(text[:max_ctx_chars])
        if not contexts:
            continue
        samples.append(
            SingleTurnSample(
                user_input=q["text"],
                retrieved_contexts=contexts,
                response=answer,
            )
        )
        query_ids.append(qid)
    if not samples:
        raise ValueError("No valid samples built — check retrieval/generation outputs.")
    return EvaluationDataset(samples=samples), query_ids


def run_ragas_eval(
    queries,
    retrieved_map,
    answers_map,
    corpus_lookup,
    llm,
    *,
    dataset_name="financebench",
    config_name="baseline",
    model_name="deepseek/deepseek-chat",
    n_samples=20,
    sample_seed=42,
):
    from ragas import RunConfig, evaluate

    eligible = [
        q for q in queries
        if q["_id"] in retrieved_map
        and q["_id"] in answers_map
        and answers_map.get(q["_id"], "").strip()
    ]
    if not eligible:
        raise ValueError("No eligible queries — run retrieval + generation first.")

    random.seed(sample_seed)
    sample = random.sample(eligible, min(n_samples, len(eligible)))
    logger.info(
        "RAGAS evaluation | dataset=%s config=%s samples=%d/%d",
        dataset_name, config_name, len(sample), len(queries),
    )

    eval_dataset, _ = _build_dataset(sample, retrieved_map, answers_map, corpus_lookup)
    faithfulness_m, answer_relevancy_m, context_utilization_m = _build_metrics(llm)

    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness_m, answer_relevancy_m, context_utilization_m],
        run_config=RunConfig(timeout=600, max_retries=5, max_wait=120),
    )
    df = result.to_pandas()
    return RAGASResult(
        dataset=dataset_name,
        config_name=config_name,
        model=model_name,
        n_samples=len(df),
        faithfulness=float(df["faithfulness"].mean()),
        answer_relevancy=float(df["answer_relevancy"].mean()),
        context_utilization=float(df["context_utilization"].mean()),
        per_query_df=df,
    )
