"""Portable configuration for the MythLens medical RAG module."""

from __future__ import annotations

import os
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
VECTOR_PATH = Path(os.getenv("MYTHLENS_VECTOR_PATH", str(RAG_DIR / "vector_store"))).expanduser().resolve()

FAISS_INDEX_PATH = Path(
    os.getenv("MYTHLENS_FAISS_INDEX_PATH", str(VECTOR_PATH / "medical_faiss.index"))
).expanduser().resolve()
METADATA_PATH = Path(
    os.getenv("MYTHLENS_METADATA_PATH", str(VECTOR_PATH / "chunk_metadata.pkl"))
).expanduser().resolve()

EMBEDDING_MODEL_NAME = os.getenv("MYTHLENS_EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")
RERANKER_MODEL_NAME = os.getenv("MYTHLENS_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

DEFAULT_TOP_K = int(os.getenv("MYTHLENS_TOP_K", "5"))
LOCAL_CANDIDATES_K = int(os.getenv("MYTHLENS_LOCAL_CANDIDATES_K", "20"))
PUBMED_CANDIDATES_K = int(os.getenv("MYTHLENS_PUBMED_CANDIDATES_K", "20"))
FAISS_CANDIDATE_K = int(os.getenv("MYTHLENS_FAISS_CANDIDATE_K", "50"))
RRF_K = int(os.getenv("MYTHLENS_RRF_K", "60"))
