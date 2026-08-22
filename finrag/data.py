"""Data loaders for ICAIF-24 JSONL corpus/queries and qrels TSV."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.debug("Loaded %d records from %s", len(records), path.name)
    return records


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    path = Path(path)
    if not path.exists():
        logger.warning("Qrels file not found: %s — evaluation will be skipped", path)
        return {}
    df = pd.read_csv(path, sep="\t")
    qrels = (
        df.groupby("query_id")
        .apply(lambda x: dict(zip(x["corpus_id"], x["score"])), include_groups=False)
        .to_dict()
    )
    logger.debug("Loaded qrels for %d queries from %s", len(qrels), path.name)
    return qrels


def make_corpus_lookup(corpus: list[dict]) -> dict[str, dict]:
    """Map corpus_id -> full document dict (with title/text)."""
    return {doc["_id"]: doc for doc in corpus}
