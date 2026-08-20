"""Prepare and score a manually judged Hybrid RAG evaluation set.

Workflow:
  1) Prepare a review file with Top-N evidence per query:
       python scripts/evaluate_hybrid_rag.py --prepare

  2) Open outputs/hybrid_eval/review.json and set each result's relevance:
       0 = irrelevant
       1 = partially/supportively relevant
       2 = directly relevant

  3) Score the reviewed file:
       python scripts/evaluate_hybrid_rag.py --score outputs/hybrid_eval/review.json

This evaluates the production mixed-source retriever fairly because both PubMed
and local evidence receive human relevance judgments rather than requiring every
result to match a PubMed PMID.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.evaluation.metrics import evaluate_graded_retrieval
from backend.rag.retriever import retrieve_evidence

DEFAULT_QUERIES = ROOT / "data" / "hybrid_eval_queries.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "hybrid_eval"


def evidence_id(item: dict[str, Any], rank: int) -> str:
    pmid = str(item.get("pmid") or "").strip()
    if pmid:
        return f"PMID:{pmid}"
    source = str(item.get("source") or "local").strip()
    title = str(item.get("title") or "").strip()
    if title:
        return f"{source}:{title}"
    return f"{source}:rank-{rank}"


def load_queries(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Query file must contain a JSON list")
    queries = []
    for index, item in enumerate(payload, 1):
        if isinstance(item, str):
            query = item.strip()
            query_id = f"q{index:02d}"
        elif isinstance(item, dict):
            query = str(item.get("query", "")).strip()
            query_id = str(item.get("id") or f"q{index:02d}")
        else:
            continue
        if query:
            queries.append({"id": query_id, "query": query})
    return queries


def prepare_review(query_file: Path, output_dir: Path, pool_k: int) -> Path:
    if pool_k < 5:
        raise ValueError("pool-k must be at least 5")

    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "review.json"
    queries = load_queries(query_file)
    review_items = []

    for query_item in queries:
        query = query_item["query"]
        print(f"Retrieving {query_item['id']}: {query}")
        result = retrieve_evidence(query, top_k=pool_k)
        evidence = result.get("evidence", [])
        candidates = []
        seen_ids: set[str] = set()
        for item in evidence:
            candidate_id = evidence_id(item, len(candidates) + 1)
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            rank = len(candidates) + 1
            candidates.append({
                "rank": rank,
                "id": candidate_id,
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "pmid": item.get("pmid", ""),
                "study_type": item.get("study_type", ""),
                "score": item.get("score", 0.0),
                "url": item.get("url", ""),
                "text_preview": str(item.get("text", ""))[:700],
                "relevance": None,
                "review_note": "",
            })
        review_items.append({
            "id": query_item["id"],
            "query": query,
            "pool_k": pool_k,
            "candidates": candidates,
        })

    payload = {
        "instructions": {
            "relevance_0": "Irrelevant: does not meaningfully answer the medical query.",
            "relevance_1": "Partially relevant: useful context/support but indirect, limited, or not the exact population/intervention/outcome.",
            "relevance_2": "Directly relevant: directly addresses the medical query with usable medical evidence.",
            "review_rule": "Judge relevance from title/abstract text and source. Do not change labels to improve metrics.",
        },
        "queries": review_items,
    }
    review_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return review_path


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-ranked occurrence of each evidence ID."""
    deduped = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("id", ""))
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        deduped.append(candidate)
    return deduped


def score_review(review_path: Path, output_dir: Path, top_k: int) -> Path:
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    query_items = payload.get("queries", [])
    metric_items = []
    per_query = []
    unreviewed = []

    for query_item in query_items:
        candidates = _dedupe_candidates(query_item.get("candidates", []))
        missing = [candidate.get("rank") for candidate in candidates if candidate.get("relevance") is None]
        if missing:
            unreviewed.append({"id": query_item.get("id"), "missing_ranks": missing})
            continue

        retrieved_ids = [str(candidate.get("id", "")) for candidate in candidates]
        relevance_by_id = {
            str(candidate.get("id", "")): int(candidate.get("relevance", 0))
            for candidate in candidates
        }
        judged_relevant_total = sum(1 for value in relevance_by_id.values() if value > 0)
        metric_item = {
            "retrieved_ids": retrieved_ids,
            "relevance_by_id": relevance_by_id,
            "judged_relevant_total": judged_relevant_total,
            "answerable": True,
        }
        one = evaluate_graded_retrieval([metric_item], top_k=top_k)
        per_query.append({
            "id": query_item.get("id"),
            "query": query_item.get("query"),
            "metrics": one,
            "retrieved_unique": len(retrieved_ids),
            "relevant_in_judged_pool": judged_relevant_total,
        })
        metric_items.append(metric_item)

    if unreviewed:
        raise ValueError(
            "Review is incomplete. Set relevance=0/1/2 for every candidate. "
            f"Missing: {json.dumps(unreviewed, ensure_ascii=False)}"
        )

    aggregate = evaluate_graded_retrieval(metric_items, top_k=top_k)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "hybrid_rag_metrics.json"
    result = {
        "evaluation_type": "manual graded relevance over Hybrid RAG candidate pool",
        "relevance_scale": {"0": "irrelevant", "1": "partial", "2": "direct"},
        "top_k": top_k,
        "judged_pool": "Unique top candidates stored in review.json for each query",
        "aggregate": aggregate,
        "per_query": per_query,
        "limitations": [
            "Recall is Recall@K over the manually judged candidate pool, not the entire PubMed/local corpus.",
            "Queries with no retrieved evidence are counted as zero-quality retrievals and reduce retrieval coverage.",
            "Duplicate chunks with the same evidence ID are deduplicated before scoring.",
            "Human relevance labels should be assigned before inspecting aggregate scores.",
            "This evaluates retrieval relevance; final verdict correctness requires a separate labeled claim-verdict set.",
        ],
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Retrieve candidates and create a manual review file")
    mode.add_argument("--score", type=Path, metavar="REVIEW_JSON", help="Score a completed review file")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pool-k", type=int, default=10, help="Number of candidates to judge per query")
    parser.add_argument("--top-k", type=int, default=5, help="K used for Precision/Recall/nDCG")
    args = parser.parse_args()

    if args.prepare:
        path = prepare_review(args.queries, args.output_dir, args.pool_k)
        print(f"\nReview file created: {path}")
        print("Set relevance to 0, 1, or 2 for every candidate, then run:")
        print(f"python scripts/evaluate_hybrid_rag.py --score {path}")
        return 0

    path = score_review(args.score, args.output_dir, args.top_k)
    result = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"result": str(path), "aggregate": result["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
