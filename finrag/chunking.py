"""Chunking strategies backed by LangChain document types/splitters.

Default: dataset-aware (passage / tabular separators).
Alternatives: fixed, semantic, tfidf-style.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkResult:
    texts: list[str]
    ids: list[str]
    original_ids: list[str]
    lc_docs: list[Document]  # Document(page_content, metadata={"id": ...})


def _merge_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    merged: list[str] = []
    for i, c in enumerate(chunks):
        if i == 0:
            merged.append(c)
            continue
        prev = chunks[i - 1]
        tail = prev[-overlap:] if overlap > 0 else ""
        merged.append(tail + c)
    return merged


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def _semantic_chunks(text: str, size: int, overlap: int) -> list[str]:
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if not current:
            current = sent
        elif len(current) + len(sent) + 1 <= size:
            current += " " + sent
        else:
            chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return _merge_overlap(chunks, overlap)


def _tfidf_style_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Approximate topically distinct sections by paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:
        return _semantic_chunks(text, size, overlap)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= size:
            current += "\n\n" + para
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return _merge_overlap(chunks, overlap)


def split_documents(
    corpus: list[dict],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    dataset_type: str = "passage",
    strategy: str = "dataset-aware",
) -> ChunkResult:
    """Chunk a corpus using the selected strategy and return LangChain Documents."""
    texts: list[str] = []
    ids: list[str] = []
    original_ids: list[str] = []
    lc_docs: list[Document] = []

    for doc_idx, doc in enumerate(corpus):
        full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()

        if strategy == "fixed":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[" "],
            )
            chunks = splitter.split_text(full_text)
        elif strategy == "semantic":
            chunks = _semantic_chunks(full_text, chunk_size, chunk_overlap)
        elif strategy == "tfidf":
            chunks = _tfidf_style_chunks(full_text, chunk_size, chunk_overlap)
        else:  # dataset-aware (default)
            separators = (
                ["\n\n", "\n", "|", " ", ""]
                if dataset_type == "tabular"
                else ["\n\n", "\n", ". ", " ", ""]
            )
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
            chunks = splitter.split_text(full_text)

        for chunk_idx, chunk in enumerate(chunks):
            cid = f"{doc_idx}_{doc['_id']}_chunk{chunk_idx}"
            ids.append(cid)
            texts.append(chunk)
            original_ids.append(doc["_id"])
            lc_docs.append(Document(page_content=chunk, metadata={"id": doc["_id"]}))

    logger.info(
        "Chunked %d docs -> %d chunks (size=%d, overlap=%d, type=%s, strategy=%s)",
        len(corpus), len(texts), chunk_size, chunk_overlap, dataset_type, strategy,
    )
    return ChunkResult(texts=texts, ids=ids, original_ids=original_ids, lc_docs=lc_docs)
