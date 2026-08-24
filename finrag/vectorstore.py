"""ChromaDB vector store via langchain_chroma.Chroma."""
from __future__ import annotations

import logging
import re
import shutil

from langchain_chroma import Chroma

from . import config
from .chunking import ChunkResult

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_").lower()


def get_vectorstore(
    dataset_name: str,
    chunk_result: ChunkResult,
    embedding_model,
    *,
    force_rebuild: bool = False,
    model_name: str | None = None,
) -> Chroma:
    """Return a persisted langchain_chroma.Chroma store for a dataset.

    The persist directory is keyed by embedding model (when known) because
    stores built with models of different dimensionality are incompatible.
    """
    if model_name:
        persist_dir = config.CHROMA_DIR / f"{dataset_name}__{_model_slug(model_name)}"
    else:
        persist_dir = config.CHROMA_DIR / dataset_name

    if force_rebuild and persist_dir.exists():
        logger.info("Force rebuild: removing %s ...", persist_dir)
        shutil.rmtree(persist_dir)

    if persist_dir.exists() and not force_rebuild:
        logger.info("Loading cached vector store for '%s' from %s", dataset_name, persist_dir)
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedding_model,
        )

    logger.info("Building vector store for '%s' (%d chunks) ...", dataset_name, len(chunk_result.texts))
    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
    )

    for i in range(0, len(chunk_result.texts), _BATCH_SIZE):
        vectorstore.add_texts(
            texts=chunk_result.texts[i : i + _BATCH_SIZE],
            metadatas=[{"id": oid} for oid in chunk_result.original_ids[i : i + _BATCH_SIZE]],
            ids=chunk_result.ids[i : i + _BATCH_SIZE],
        )

    logger.info("Vector store for '%s' built and persisted.", dataset_name)
    return vectorstore
