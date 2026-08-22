"""Retrieval evaluation — mean NDCG@k."""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def _dcg(rels: list[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))


def compute_ndcg(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k: int = 10,
) -> float:
    ndcg_scores: list[float] = []
    total_labeled = sum(1 for qid in qrels if qid in results)

    for qid, doc_scores in results.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        top = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        true = [relevant.get(doc_id, 0) for doc_id, _ in top]
        if not top or sum(true) == 0:
            continue
        ndcg = _dcg(true, k) / max(_dcg(sorted(true, reverse=True), k), 1e-9)
        ndcg_scores.append(ndcg)

    if not ndcg_scores:
        logger.warning("No scoreable queries found — check qrels alignment.")
        return float("nan")

    covered = len(ndcg_scores)
    logger.info(
        "Coverage: %d / %d queries with >=1 relevant doc in top-%d (%.1f%%)",
        covered, total_labeled, k, 100 * covered / max(total_labeled, 1),
    )
    return sum(ndcg_scores) / len(ndcg_scores)
