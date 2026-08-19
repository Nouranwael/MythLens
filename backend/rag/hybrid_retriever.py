"""Hybrid local retrieval using FAISS + BM25 with reciprocal-rank fusion."""
from .config import RRF_K
from .vector_store import vector_search
from .bm25_retriever import bm25_search


def hybrid_search(query, top_k=10, candidate_k=30, rrf_k=RRF_K):
    vector_results = vector_search(query, top_k=candidate_k)
    bm25_results = bm25_search(query, top_k=candidate_k)
    combined = {}

    for rank, item in enumerate(vector_results, start=1):
        item_id = item.get("id", f"vector-{rank}")
        combined.setdefault(item_id, {**item, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None})
        combined[item_id]["vector_rank"] = rank
        combined[item_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, item in enumerate(bm25_results, start=1):
        item_id = item.get("id", f"bm25-{rank}")
        combined.setdefault(item_id, {**item, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None})
        combined[item_id]["bm25_rank"] = rank
        combined[item_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    return sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]
