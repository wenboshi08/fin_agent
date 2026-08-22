#!/usr/bin/env python3
"""Synthetic local demo for FinAgent MVP.

Shows the full flow: chunking -> retrieval (dense-like + BM25-like + RRF + MMR)
-> grounded generation (fake LLM, no API) -> NDCG evaluation.

This script is intentionally dependency-free (stdlib only) so it can run in a
plain environment without Python 3.12, LangChain, models, or API keys.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

# Allow `python scripts/demo_synthetic.py` from anywhere inside the repo.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrag.evaluation import compute_ndcg  # stdlib-only module


# ── Fake data ────────────────────────────────────────────────────────────────
CORPUS = [
    {
        "_id": "doc_a",
        "title": "Acme FY2024 Annual Report",
        "text": (
            "Acme Inc. total revenue in fiscal year 2024 was $10.2 billion. "
            "Gross profit was $4.1 billion. Operating expenses were $3.0 billion. "
            "Net income was $1.2 billion. Cash and cash equivalents at year end were $5.5 billion."
        ),
    },
    {
        "_id": "doc_b",
        "title": "Acme FY2023 Annual Report",
        "text": (
            "Acme Inc. total revenue in fiscal year 2023 was $8.8 billion. "
            "Net income was $0.9 billion. "
            "The company noted strong demand in the enterprise segment."
        ),
    },
    {
        "_id": "doc_c",
        "title": "Globex Q3 2024 Report",
        "text": (
            "Globex Corp. third quarter revenue increased 15% year over year to $2.5 billion. "
            "Management attributed growth to strong enterprise demand and new product launches."
        ),
    },
]

QUERIES = [
    {"_id": "q1", "text": "What was Acme's total revenue in FY2024?"},
    {"_id": "q2", "text": "What was Acme's net income in FY2023?"},
    {"_id": "q3", "text": "What drove Globex Q3 revenue growth?"},
]

QRELS = {
    "q1": {"doc_a": 1},
    "q2": {"doc_b": 1},
    "q3": {"doc_c": 1},
}


# ── Step 1: chunking ─────────────────────────────────────────────────────────
def chunk_documents(corpus: list[dict]) -> list[dict]:
    chunks = []
    for doc in corpus:
        full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        # Simple sentence-level chunking for demo purposes.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if s.strip()]
        for i, sent in enumerate(sentences):
            chunks.append(
                {
                    "chunk_id": f"{doc['_id']}_chunk{i}",
                    "doc_id": doc["_id"],
                    "text": sent,
                }
            )
    return chunks


# ── Step 2: two fake retrievers ─────────────────────────────────────────────
def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9$%]+", text.lower()))


def bm25_like(query: str, chunks: list[dict]) -> dict[str, float]:
    """Keyword-ish scorer: count shared tokens plus slight rarity boost."""
    q_tokens = tokenize(query)
    scores = {}
    for chunk in chunks:
        c_tokens = tokenize(chunk["text"])
        overlap = q_tokens & c_tokens
        score = sum(1.0 / math.log(2 + len(c_tokens)) for _ in overlap)
        scores[chunk["doc_id"]] = scores.get(chunk["doc_id"], 0.0) + score
    return scores


def dense_like(query: str, chunks: list[dict]) -> dict[str, float]:
    """Semantic-ish scorer: fake embedding cosine using character bigrams.

    Uses the best matching chunk per doc (max), avoiding a multi-chunk doc
    being unfairly boosted by simply having more chunks.
    """
    def bigrams(text: str) -> dict[str, int]:
        t = re.sub(r"[^a-z0-9]", "", text.lower())
        grams = {}
        for i in range(len(t) - 1):
            g = t[i : i + 2]
            grams[g] = grams.get(g, 0) + 1
        return grams

    q_vec = bigrams(query)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
    by_doc: dict[str, list[float]] = {}
    for chunk in chunks:
        c_vec = bigrams(chunk["text"])
        dot = sum(q_vec.get(g, 0) * v for g, v in c_vec.items())
        c_norm = math.sqrt(sum(v * v for v in c_vec.values()))
        sim = dot / (q_norm * c_norm) if q_norm and c_norm else 0.0
        by_doc.setdefault(chunk["doc_id"], []).append(sim)

    return {doc_id: max(sims) for doc_id, sims in by_doc.items()}


# ── Step 3: RRF fusion ──────────────────────────────────────────────────────
def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Step 4: MMR-like diversity selection ────────────────────────────────────
def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def mmr_select(
    candidates: list[str],
    relevance: dict[str, float],
    chunks_by_doc: dict[str, list[str]],
    k: int = 3,
    lam: float = 0.7,
) -> list[str]:
    remaining = list(candidates)
    selected: list[str] = []
    while len(selected) < k and remaining:
        best_id = None
        best_score = -1e9
        for doc_id in remaining:
            rel = relevance.get(doc_id, 0.0)
            if selected:
                doc_tokens = set()
                for text in chunks_by_doc.get(doc_id, []):
                    doc_tokens |= tokenize(text)
                max_sim = 0.0
                for sel_id in selected:
                    sel_tokens = set()
                    for text in chunks_by_doc.get(sel_id, []):
                        sel_tokens |= tokenize(text)
                    max_sim = max(max_sim, jaccard(doc_tokens, sel_tokens))
            else:
                max_sim = 0.0
            score = lam * rel - (1 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_id = doc_id
        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


# ── Step 5: fake generation (no LLM) ────────────────────────────────────────
def fake_generate(query: str, top_doc: str, best_snippet: str) -> str:
    return (
        f"[synthetic answer, no LLM] According to {top_doc}: "
        f"\"{best_snippet}\" [citation: {top_doc}]"
    )


# ── Main demo ───────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("FinAgent Synthetic Demo (no API / no models / no LangChain)")
    print("=" * 70)

    # 1. Chunk
    chunks = chunk_documents(CORPUS)
    print(f"\n[1] Chunking")
    print(f"    {len(CORPUS)} docs -> {len(chunks)} sentence chunks")
    for c in chunks[:3]:
        print(f"      {c['chunk_id']}: {c['text'][:60]}...")

    chunks_by_doc: dict[str, list[str]] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c["doc_id"], []).append(c["text"])

    results: dict[str, dict[str, float]] = {}

    # 2-6. Per-query pipeline
    for q in QUERIES:
        print(f"\n[2-6] Query: {q['text']}")

        # two retrievers
        bm25_scores = bm25_like(q["text"], chunks)
        dense_scores = dense_like(q["text"], chunks)

        bm25_ranked = [d for d, _ in sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)]
        dense_ranked = [d for d, _ in sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)]
        print(f"    BM25-like : {bm25_ranked}")
        print(f"    Dense-like: {dense_ranked}")

        # RRF
        fused = rrf_fuse([dense_ranked, bm25_ranked])
        fused_ids = [doc_id for doc_id, _ in fused]
        print(f"    RRF fused : {fused_ids[:4]}")

        # MMR
        selected = mmr_select(
            fused_ids,
            dense_scores,
            chunks_by_doc,
            k=3,
            lam=0.7,
        )
        print(f"    MMR select: {selected}")

        # Pick best snippet from top doc
        top_doc = selected[0]
        best_snippet = max(chunks_by_doc[top_doc], key=lambda t: len(tokenize(q["text"]) & tokenize(t)))
        answer = fake_generate(q["text"], top_doc, best_snippet)
        print(f"    Answer    : {answer}")

        # Store top-k scores for NDCG
        results[q["_id"]] = {
            doc_id: max(0.0, 1.0 - i / 10.0)
            for i, doc_id in enumerate(selected)
        }

    # 7. Evaluate
    print("\n[7] Evaluation")
    ndcg = compute_ndcg(QRELS, results, k=3)
    print(f"    NDCG@3 = {ndcg:.4f} (1.0 = perfect ranking)")

    print("\nDone. This mirrors the real pipeline but uses fake scores instead of embeddings/LLMs.")


if __name__ == "__main__":
    main()
