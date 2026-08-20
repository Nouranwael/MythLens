"""Evaluation metrics for verdict classification and evidence retrieval."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def evaluate_verdicts(predictions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same length")
    if not ground_truth:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "detailed_report": {}}
    accuracy = float(accuracy_score(ground_truth, predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="weighted", zero_division=0
    )
    report = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "detailed_report": report,
    }


def evaluate_retrieval(retrieval_results: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, float]:
    """Legacy exact-ID retrieval evaluation.

    This is useful when a complete gold set of document/chunk IDs already exists.
    For mixed-source Hybrid RAG evaluation, prefer ``evaluate_graded_retrieval``.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    precisions, recalls, reciprocal_ranks = [], [], []
    for item in retrieval_results:
        retrieved = item.get("retrieved_ids", [])[:top_k]
        relevant = set(item.get("relevant_ids", []))
        if not relevant:
            continue
        hits = [1 if doc_id in relevant else 0 for doc_id in retrieved]
        precisions.append(sum(hits) / top_k)
        recalls.append(sum(hits) / len(relevant))
        reciprocal_ranks.append(next((1.0 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0))
    return {
        f"precision@{top_k}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f"recall@{top_k}": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "mrr": round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 4),
    }


def _dcg(relevances: List[int], top_k: int) -> float:
    score = 0.0
    for rank, relevance in enumerate(relevances[:top_k], start=1):
        score += (2 ** int(relevance) - 1) / math.log2(rank + 1)
    return score


def evaluate_graded_retrieval(retrieval_results: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
    """Evaluate manually judged Hybrid RAG results.

    Expected item format::

        {
          "retrieved_ids": ["doc-a", "doc-b", ...],
          "relevance_by_id": {"doc-a": 2, "doc-b": 0, ...},
          "judged_relevant_total": 4
        }

    Relevance grades:
      0 = irrelevant
      1 = partially/supportively relevant
      2 = directly relevant to the medical query

    Recall is calculated against the complete *judged candidate pool* supplied for
    each query, not against the entire corpus. This limitation is reported in the
    output so the metric is not overclaimed.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    precisions: List[float] = []
    recalls: List[float] = []
    reciprocal_ranks: List[float] = []
    ndcgs: List[float] = []
    skipped = 0

    for item in retrieval_results:
        retrieved = [str(value) for value in item.get("retrieved_ids", [])]
        relevance_by_id = {
            str(key): int(value)
            for key, value in item.get("relevance_by_id", {}).items()
            if value is not None
        }
        judged_total = int(item.get("judged_relevant_total", 0) or 0)

        if not retrieved or not relevance_by_id:
            skipped += 1
            continue

        top = retrieved[:top_k]
        grades = [max(0, min(2, relevance_by_id.get(doc_id, 0))) for doc_id in top]
        hits = [1 if grade > 0 else 0 for grade in grades]

        precisions.append(sum(hits) / top_k)
        recalls.append((sum(hits) / judged_total) if judged_total > 0 else 0.0)
        reciprocal_ranks.append(next((1.0 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0))

        ideal_grades = sorted(relevance_by_id.values(), reverse=True)
        dcg = _dcg(grades, top_k)
        idcg = _dcg(ideal_grades, top_k)
        ndcgs.append((dcg / idcg) if idcg > 0 else 0.0)

    return {
        f"precision@{top_k}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f"recall@{top_k}": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "mrr": round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 4),
        f"ndcg@{top_k}": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        "evaluated_queries": len(precisions),
        "skipped_queries": skipped,
        "recall_scope": "judged candidate pool",
    }
