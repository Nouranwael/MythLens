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

    Relevance grades:
      0 = irrelevant
      1 = partially/supportively relevant
      2 = directly relevant to the medical query

    Standard relevance treats both grades 1 and 2 as relevant. nDCG keeps the
    full 0/1/2 grading so directly relevant evidence receives more gain.

    Empty retrievals are counted as zero-quality results when ``answerable`` is
    true (the default), rather than silently skipped. This prevents retrieval
    failures from inflating aggregate metrics.

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
    no_result_queries = 0
    queries_with_results = 0

    for item in retrieval_results:
        retrieved = [str(value) for value in item.get("retrieved_ids", []) if str(value)]
        relevance_by_id = {
            str(key): int(value)
            for key, value in item.get("relevance_by_id", {}).items()
            if value is not None
        }
        judged_total = int(item.get("judged_relevant_total", 0) or 0)
        answerable = bool(item.get("answerable", True))

        if not retrieved:
            if answerable:
                no_result_queries += 1
                precisions.append(0.0)
                recalls.append(0.0)
                reciprocal_ranks.append(0.0)
                ndcgs.append(0.0)
            else:
                skipped += 1
            continue

        queries_with_results += 1
        if not relevance_by_id:
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

    evaluated = len(precisions)
    coverage_denominator = queries_with_results + no_result_queries
    coverage = (queries_with_results / coverage_denominator) if coverage_denominator else 0.0

    return {
        f"precision@{top_k}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f"recall@{top_k}": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "mrr": round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 4),
        f"ndcg@{top_k}": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        "retrieval_coverage": round(float(coverage), 4),
        "evaluated_queries": evaluated,
        "queries_with_results": queries_with_results,
        "no_result_queries": no_result_queries,
        "skipped_queries": skipped,
        "recall_scope": "judged candidate pool",
    }


def evaluate_strict_retrieval(retrieval_results: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
    """Strict retrieval evaluation where only grade 2 counts as relevant.

    Precision and MRR are computed over every evaluated query, including queries
    whose judged pool contains no grade-2 evidence. Strict Recall is undefined for
    those queries, so they are excluded from the recall mean and reported through
    ``recall_evaluable_queries``.

    ``strict_ndcg`` uses binary direct relevance (2 -> 1, 0/1 -> 0), complementing
    the standard graded nDCG that rewards both partial and direct relevance.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    precisions: List[float] = []
    recalls: List[float] = []
    reciprocal_ranks: List[float] = []
    ndcgs: List[float] = []
    skipped = 0
    no_result_queries = 0
    queries_with_results = 0
    queries_with_direct_in_pool = 0
    queries_with_direct_in_top_k = 0

    for item in retrieval_results:
        retrieved = [str(value) for value in item.get("retrieved_ids", []) if str(value)]
        relevance_by_id = {
            str(key): int(value)
            for key, value in item.get("relevance_by_id", {}).items()
            if value is not None
        }
        answerable = bool(item.get("answerable", True))

        if not retrieved:
            if answerable:
                no_result_queries += 1
                precisions.append(0.0)
                reciprocal_ranks.append(0.0)
                ndcgs.append(0.0)
            else:
                skipped += 1
            continue

        queries_with_results += 1
        if not relevance_by_id:
            skipped += 1
            continue

        top = retrieved[:top_k]
        strict_hits = [1 if relevance_by_id.get(doc_id, 0) == 2 else 0 for doc_id in top]
        direct_total = sum(1 for grade in relevance_by_id.values() if grade == 2)

        precisions.append(sum(strict_hits) / top_k)
        reciprocal_ranks.append(next((1.0 / rank for rank, hit in enumerate(strict_hits, 1) if hit), 0.0))

        if any(strict_hits):
            queries_with_direct_in_top_k += 1

        if direct_total > 0:
            queries_with_direct_in_pool += 1
            recalls.append(sum(strict_hits) / direct_total)
            ideal = [1] * min(direct_total, top_k)
            ideal.extend([0] * (top_k - len(ideal)))
            dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(strict_hits, 1))
            idcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(ideal, 1))
            ndcgs.append((dcg / idcg) if idcg > 0 else 0.0)
        else:
            ndcgs.append(0.0)

    evaluated = len(precisions)
    return {
        f"strict_precision@{top_k}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f"strict_recall@{top_k}": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "strict_mrr": round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 4),
        f"strict_ndcg@{top_k}": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        f"direct_relevance_coverage@{top_k}": round(
            (queries_with_direct_in_top_k / evaluated) if evaluated else 0.0, 4
        ),
        "evaluated_queries": evaluated,
        "recall_evaluable_queries": len(recalls),
        "queries_with_direct_evidence_in_pool": queries_with_direct_in_pool,
        "queries_with_direct_evidence_in_top_k": queries_with_direct_in_top_k,
        "no_result_queries": no_result_queries,
        "skipped_queries": skipped,
        "relevance_rule": "only grade 2 is relevant",
        "recall_scope": "grade-2 evidence in judged candidate pool",
    }
