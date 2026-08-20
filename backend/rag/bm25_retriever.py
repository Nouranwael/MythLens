"""BM25 keyword retrieval over Member 2's local evidence metadata."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from .vector_store import get_chunked_records, local_assets_available


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", str(text).lower())


@lru_cache(maxsize=1)
def _build_bm25():
    if not local_assets_available():
        return None, []
    from rank_bm25 import BM25Okapi
    records = get_chunked_records()
    corpus = [tokenize(record.get("text", "")) for record in records]
    return BM25Okapi(corpus), records


def bm25_search(query, top_k=10):
    bm25, records = _build_bm25()
    if bm25 is None or not records:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        record = records[int(idx)].copy()
        record["bm25_score"] = float(scores[int(idx)])
        results.append(record)
    return results
