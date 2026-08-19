"""Evidence scoring and final reranking for local and PubMed candidates."""
from __future__ import annotations

from functools import lru_cache
import math

from .config import RERANKER_MODEL_NAME
from .vector_store import get_embedding_model


def study_strength(publication_types):
    types_text = " ".join(publication_types or []).lower()
    if "meta-analysis" in types_text: return 1.00
    if "systematic review" in types_text: return 0.95
    if "randomized controlled trial" in types_text: return 0.90
    if "clinical trial" in types_text: return 0.85
    if "review" in types_text: return 0.80
    if "observational study" in types_text: return 0.70
    if "case reports" in types_text or "case report" in types_text: return 0.50
    if "journal article" in types_text: return 0.65
    return 0.60


def score_pubmed_articles(query, articles):
    if not articles:
        return []
    try:
        model = get_embedding_model()
        query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        texts = [(a.get("title", "") + " " + a.get("text", "")).strip() for a in articles]
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        similarities = embeddings @ query_embedding
        return [{**article, "semantic_score": float(score)} for article, score in zip(articles, similarities)]
    except Exception:
        return [{**article, "semantic_score": 0.0} for article in articles]


def rerank_pubmed_articles(articles):
    results = []
    for article in articles:
        strength = study_strength(article.get("publication_types", []))
        semantic = float(article.get("semantic_score", 0.0))
        results.append({**article, "study_strength": strength, "source_score": 0.80 * semantic + 0.20 * strength})
    return sorted(results, key=lambda x: x["source_score"], reverse=True)


@lru_cache(maxsize=1)
def _get_cross_encoder():
    import torch
    from sentence_transformers import CrossEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(RERANKER_MODEL_NAME, device=device)


def _sigmoid(value):
    try:
        return 1.0 / (1.0 + math.exp(-float(value)))
    except OverflowError:
        return 0.0 if float(value) < 0 else 1.0


def final_rerank(query, candidates, top_k=5, min_relevance=0.50):
    if not candidates:
        return []
    pairs = [(query, (c.get("title", "") + " " + c.get("text", "")).strip()) for c in candidates]
    try:
        raw_scores = _get_cross_encoder().predict(pairs, show_progress_bar=False)
        relevance_scores = [_sigmoid(score) for score in raw_scores]
    except Exception:
        relevance_scores = [max(0.0, min(1.0, float(c.get("source_score", c.get("rrf_score", 0.6))))) for c in candidates]

    final_results = []
    for candidate, relevance in zip(candidates, relevance_scores):
        item = candidate.copy()
        item["rerank_score"] = float(relevance)
        if item.get("source") == "PubMed":
            final_score = 0.70 * relevance + 0.20 * float(item.get("study_strength", 0.60)) + 0.10 * float(item.get("semantic_score", 0.0))
        else:
            final_score = relevance
        item["final_score"] = float(final_score)
        if relevance >= min_relevance:
            final_results.append(item)

    return sorted(final_results, key=lambda x: x["final_score"], reverse=True)[:top_k]
