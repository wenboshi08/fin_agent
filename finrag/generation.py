"""Grounded generation with citations using LangChain + DeepSeek/Qwen."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a precise financial analyst assistant with expertise in SEC filings, \
earnings reports, and financial disclosures.

Rules:
- Answer using ONLY the document excerpts provided. Do not use outside knowledge.
- Be specific: quote exact figures, percentages, and fiscal periods from the context.
- For numerical questions, show your reasoning step-by-step.
- Cite every document you use with its ID in square brackets, e.g. [dd2af2336].
- If the context does not contain enough information, respond with exactly: \
"Insufficient information in the provided documents."
"""

_HUMAN = """\
Document excerpts:
{context}

---
Question: {query}
"""


@dataclass
class CitedSource:
    corpus_id: str
    title: str
    score: float
    excerpt: str


@dataclass
class GenerationResult:
    query: str
    answer: str
    sources: list[CitedSource]
    model: str = "unknown"
    multiquery: bool = False

    def pretty(self) -> str:
        lines = [
            "=" * 60,
            f"Query : {self.query}",
            "=" * 60,
            f"\nAnswer\n{self.answer}\n",
        ]
        if self.sources:
            lines.append("Sources")
            for i, src in enumerate(self.sources, 1):
                title = f" — {src.title}" if src.title else ""
                lines.append(f" [{i}] {src.corpus_id}{title}")
                lines.append(f" score={src.score:.3f}")
            lines.append("=" * 60)
        return "\n".join(lines)


def _build_context(
    retrieved: list[tuple[str, float]],
    corpus_lookup: dict[str, dict],
    max_chars: int = 1_500,
) -> tuple[str, list[CitedSource]]:
    blocks: list[str] = []
    sources: list[CitedSource] = []
    for doc_id, score in retrieved:
        doc = corpus_lookup.get(doc_id)
        if doc is None:
            continue
        title = doc.get("title", "")
        text = doc.get("text", "")
        text_trimmed = text[:max_chars]
        if len(text) > max_chars:
            text_trimmed += " …[truncated]"
        header = f"[{doc_id}]" + (f" {title}" if title else "")
        blocks.append(f"{header}\n{text_trimmed}")
        sources.append(CitedSource(doc_id, title, score, text[:400]))
    return "\n\n---\n\n".join(blocks), sources


def _extract_ids_from_text(text: str, valid_ids: set[str]) -> list[str]:
    found = re.findall(r"\[([a-zA-Z0-9_]+)\]", text)
    return [fid for fid in found if fid in valid_ids]


def generate_answer(
    query: str,
    retrieved: list[tuple[str, float]],
    corpus_lookup: dict[str, dict],
    llm,
    *,
    top_k: int = 5,
    max_chars: int = 1_500,
    multiquery: bool = False,
    model_name: str | None = None,
) -> GenerationResult:
    if not retrieved:
        return GenerationResult(
            query=query,
            answer="No relevant documents were retrieved for this query.",
            sources=[],
            model=model_name or "unknown",
            multiquery=multiquery,
        )

    top_retrieved = retrieved[:top_k]
    context, sources = _build_context(top_retrieved, corpus_lookup, max_chars=max_chars)
    if not sources:
        return GenerationResult(
            query=query,
            answer="Retrieved documents could not be loaded from corpus.",
            sources=[],
            model=model_name or "unknown",
            multiquery=multiquery,
        )

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", _HUMAN)]
    )
    try:
        response = prompt.invoke({"context": context, "query": query})
        answer = llm.invoke(response.to_messages()).content
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return GenerationResult(
            query=query,
            answer=f"Generation error: {exc}",
            sources=sources,
            model=model_name or "unknown",
            multiquery=multiquery,
        )

    valid_ids = {s.corpus_id for s in sources}
    cited_set = set(_extract_ids_from_text(answer, valid_ids))
    cited_sources = (
        [s for s in sources if s.corpus_id in cited_set] if cited_set else sources
    )
    return GenerationResult(
        query=query,
        answer=answer,
        sources=cited_sources,
        model=model_name or "unknown",
        multiquery=multiquery,
    )
