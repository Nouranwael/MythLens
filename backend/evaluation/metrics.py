"""Evaluation metrics for verdict classification and evidence retrieval."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def evaluate_verdicts(predictions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same length")
    if not ground_truth:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "detailed_report": {}}
    accuracy = float(accuracy_score(ground_truth, predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average="weighted", zero_division=0)
    report = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)
    return {"accuracy": round(accuracy, 4), "precision": round(float(precision), 4), "recall": round(float(recall), 4), "f1_score": round(float(f1), 4), "detailed_report": report}


def evaluate_retrieval(retrieval_results: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, float]:
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
    return {f"precision@{top_k}": round(float(np.mean(precisions)) if precisions else 0.0, 4), f"recall@{top_k}": round(float(np.mean(recalls)) if recalls else 0.0, 4), "mrr": round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 4)}
