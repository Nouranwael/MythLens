"""Unified MythLens evidence retrieval: local hybrid search + live PubMed."""
from __future__ import annotations

from .config import DEFAULT_TOP_K, FAISS_CANDIDATE_K, LOCAL_CANDIDATES_K, PUBMED_CANDIDATES_K
from .hybrid_retriever import hybrid_search
from .pubmed_retriever import fetch_pubmed_articles, get_best_study_type, remove_retracted_articles, search_pubmed_multi_query
from .reranker import final_rerank, rerank_pubmed_articles, score_pubmed_articles


def retrieve_evidence(medical_query, top_k=DEFAULT_TOP_K):
    query = str(medical_query or "").strip()
    if not query:
        return {"evidence": []}

    local_candidates = hybrid_search(query, top_k=LOCAL_CANDIDATES_K, candidate_k=FAISS_CANDIDATE_K)
    pubmed_candidates = []

    try:
        pmids = search_pubmed_multi_query(query, retmax_per_query=PUBMED_CANDIDATES_K)
        articles = [a for a in fetch_pubmed_articles(pmids) if a.get("text", "").strip()]
        articles = remove_retracted_articles(articles)
        articles = rerank_pubmed_articles(score_pubmed_articles(query, articles))
        for article in articles:
            publication_types = article.get("publication_types", [])
            pubmed_candidates.append({
                "id": article.get("id", ""),
                "dataset": "PubMed",
                "source": "PubMed",
                "claim": query,
                "title": article.get("title", ""),
                "text": article.get("text", ""),
                "url": article.get("url", ""),
                "pmid": article.get("pmid", ""),
                "study_type": get_best_study_type(publication_types),
                "publication_types": publication_types,
                "semantic_score": article.get("semantic_score", 0.0),
                "study_strength": article.get("study_strength", 0.60),
                "source_score": article.get("source_score", 0.0),
            })
    except Exception:
        pass

    candidates = local_candidates + pubmed_candidates
    if not candidates:
        return {"evidence": []}

    final_results = final_rerank(query, candidates, top_k=top_k)
    evidence = []
    for item in final_results:
        evidence.append({
            "source": item.get("source", item.get("dataset", "")),
            "title": item.get("title", ""),
            "text": item.get("text", ""),
            "url": item.get("url", ""),
            "pmid": item.get("pmid", ""),
            "study_type": item.get("study_type", ""),
            "score": round(float(item.get("final_score", 0.0)), 4),
        })
    return {"evidence": evidence}
