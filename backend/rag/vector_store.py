"""FAISS-backed local evidence store used by MythLens retrieval.

The large FAISS/metadata assets are intentionally not committed to GitHub. If they
are missing, local retrieval is disabled gracefully and PubMed can still be used.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple

from .config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, METADATA_PATH


def local_assets_available() -> bool:
    return Path(FAISS_INDEX_PATH).exists() and Path(METADATA_PATH).exists()


@lru_cache(maxsize=1)
def _load_store() -> Tuple[Any, Any, List[dict]]:
    if not local_assets_available():
        raise FileNotFoundError(
            "Member 2 vector assets are not installed. Copy medical_faiss.index and "
            "chunk_metadata.pkl into backend/rag/vector_store/ or set MYTHLENS_VECTOR_PATH."
        )

    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "rb") as handle:
        records = pickle.load(handle)

    if index.ntotal != len(records):
        raise ValueError(
            f"FAISS vectors and metadata count do not match: index={index.ntotal}, metadata={len(records)}"
        )
    return model, index, records


def get_chunked_records() -> List[dict]:
    if not local_assets_available():
        return []
    return _load_store()[2]


def get_embedding_model():
    """Return the biomedical embedding model, loading it lazily."""
    if local_assets_available():
        return _load_store()[0]

    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


def vector_search(query: str, top_k: int = 10):
    if not local_assets_available():
        return []

    model, index, records = _load_store()
    query_embedding = model.encode(
        [str(query)], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        record = records[idx].copy()
        record["vector_score"] = float(score)
        results.append(record)
    return results
