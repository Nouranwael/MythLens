"""Evaluate MythLens RAG components and generate a visual dashboard.

Examples:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --load-models --output-dir outputs/rag_eval
    python scripts/evaluate_rag.py --ground-truth data/rag_eval.json --live-pubmed

Ground-truth format:
[
  {"query": "...", "relevant_ids": ["chunk-id", "pmid"]}
]
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.evaluation.metrics import evaluate_retrieval
from backend.rag.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    RERANKER_MODEL_NAME,
)
from backend.rag.vector_store import local_assets_available

DEFAULT_QUERIES = [
    "garlic blood pressure cardiovascular risk",
    "caffeine children sleep anxiety",
    "microwave radiation cancer risk",
    "metformin type 2 diabetes treatment",
]


def now_ms() -> float:
    return time.perf_counter() * 1000


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def describe_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "stdev": None}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
    }


def model_inventory(load_models: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "transcription": {"name": "faster-whisper tiny", "loaded": False},
        "claim_extraction": {"provider": "Groq", "model": "qwen/qwen3.6-27b", "loaded": False},
        "query_generation": {"provider": "Gemini", "model_env": "GEMINI_QUERY_MODEL or GEMINI_MODEL", "loaded": False},
        "verification": {"provider": "Gemini", "model_env": "GEMINI_VERIFIER_MODEL or GEMINI_MODEL", "loaded": False},
        "embedding": {"name": EMBEDDING_MODEL_NAME, "loaded": False},
        "reranker": {"name": RERANKER_MODEL_NAME, "loaded": False},
    }
    if not load_models:
        return result

    try:
        from backend.rag.vector_store import get_embedding_model
        start = now_ms()
        embedding_model = get_embedding_model()
        result["embedding"].update({"loaded": True, "load_ms": round(now_ms() - start, 2), "dimension": int(embedding_model.get_sentence_embedding_dimension())})
    except Exception as exc:
        result["embedding"]["error"] = f"{type(exc).__name__}: {exc}"

    try:
        from backend.rag.reranker import _get_cross_encoder
        start = now_ms()
        _get_cross_encoder()
        result["reranker"].update({"loaded": True, "load_ms": round(now_ms() - start, 2)})
    except Exception as exc:
        result["reranker"]["error"] = f"{type(exc).__name__}: {exc}"
    return result


def chunking_report() -> dict[str, Any]:
    if not local_assets_available():
        return {
            "status": "unavailable",
            "reason": "FAISS index and chunk_metadata.pkl are not present",
            "faiss_index": str(FAISS_INDEX_PATH),
            "metadata": str(METADATA_PATH),
        }
    try:
        from backend.rag.vector_store import get_chunked_records
        records = get_chunked_records()
        lengths = [len(str(record.get("text", "")).split()) for record in records if record.get("text")]
        source_counts: dict[str, int] = {}
        for record in records:
            source = str(record.get("source", record.get("dataset", "unknown")))
            source_counts[source] = source_counts.get(source, 0) + 1
        return {
            "status": "ok",
            "records": len(records),
            "token_proxy": describe_values([float(value) for value in lengths]),
            "sources": source_counts,
            "empty_text_records": sum(1 for record in records if not str(record.get("text", "")).strip()),
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def embedding_report(queries: list[str], load_models: bool) -> dict[str, Any]:
    if not load_models:
        return {"status": "skipped", "reason": "run with --load-models to measure embeddings"}
    try:
        from backend.rag.vector_store import get_embedding_model
        model = get_embedding_model()
        start = now_ms()
        vectors = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        elapsed = now_ms() - start
        norms = [float((vector * vector).sum() ** 0.5) for vector in vectors]
        similarities = []
        for index in range(len(vectors)):
            for other in range(index + 1, len(vectors)):
                similarities.append(float(vectors[index] @ vectors[other]))
        return {
            "status": "ok",
            "queries": len(queries),
            "encode_ms": round(elapsed, 2),
            "ms_per_query": round(elapsed / max(1, len(queries)), 2),
            "norms": describe_values(norms),
            "pairwise_cosine_similarity": describe_values(similarities),
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def retrieval_report(queries: list[str], ground_truth: list[dict[str, Any]] | None, live_pubmed: bool) -> dict[str, Any]:
    from backend.rag.retriever import retrieve_evidence

    samples = []
    metric_items = []
    for index, query in enumerate(queries):
        start = now_ms()
        try:
            result = retrieve_evidence(query, top_k=5) if live_pubmed else {"evidence": []}
            evidence = result.get("evidence", [])
            retrieved_ids = [str(item.get("pmid") or item.get("id") or item.get("title", "")) for item in evidence]
            elapsed = now_ms() - start
            sample = {"query": query, "latency_ms": round(elapsed, 2), "results": len(evidence), "scores": [item.get("score", 0.0) for item in evidence], "retrieved_ids": retrieved_ids}
            samples.append(sample)
            if ground_truth and index < len(ground_truth):
                metric_items.append({"retrieved_ids": retrieved_ids, "relevant_ids": ground_truth[index].get("relevant_ids", [])})
        except Exception as exc:
            samples.append({"query": query, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

    report: dict[str, Any] = {
        "status": "ok" if live_pubmed else "skipped",
        "mode": "live_pubmed_and_local" if live_pubmed else "not_run",
        "samples": samples,
        "latency_ms": describe_values([float(item["latency_ms"]) for item in samples if "latency_ms" in item]),
        "score_distribution": describe_values([float(score) for item in samples for score in item.get("scores", []) if safe_float(score) is not None]),
    }
    if metric_items:
        report["retrieval_metrics"] = evaluate_retrieval(metric_items, top_k=5)
    else:
        report["retrieval_metrics"] = {"status": "unavailable", "reason": "provide --ground-truth for Precision@K, Recall@K, and MRR"}
    return report


def plot_dashboard(report: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    figure.suptitle("MythLens RAG Evaluation Dashboard", fontsize=18, fontweight="bold")
    colors = ["#20b2aa", "#4f81bd", "#f2a65a", "#d95f59", "#7a5195"]

    model_names = ["Embedding", "Reranker"]
    load_times = [report["models"]["embedding"].get("load_ms", 0) or 0, report["models"]["reranker"].get("load_ms", 0) or 0]
    axes[0, 0].bar(model_names, load_times, color=colors[:2])
    axes[0, 0].set_title("Model load time (ms)")
    axes[0, 0].tick_params(axis="x", rotation=20)

    pairwise = report["embeddings"].get("pairwise_cosine_similarity", {})
    axes[0, 1].bar(["Mean", "Median", "Max"], [pairwise.get("mean") or 0, pairwise.get("median") or 0, pairwise.get("max") or 0], color=colors[2])
    axes[0, 1].set_title("Query cosine similarity")
    axes[0, 1].set_ylim(-1, 1)

    chunk = report["chunking"]
    token_stats = chunk.get("token_proxy", {})
    axes[0, 2].bar(["Min", "Mean", "Median", "Max"], [token_stats.get(key) or 0 for key in ("min", "mean", "median", "max")], color=colors[0])
    axes[0, 2].set_title("Chunk size proxy (words)")
    if chunk.get("status") != "ok":
        axes[0, 2].text(0.5, 0.5, "Unavailable\\nFAISS metadata missing", ha="center", va="center", transform=axes[0, 2].transAxes, color="#b33a3a", fontweight="bold")
        axes[0, 2].set_ylim(0, 1)

    retrieval = report["retrieval"]
    samples = retrieval.get("samples", [])
    latencies = [item.get("latency_ms", 0) for item in samples]
    axes[1, 0].bar(range(1, len(latencies) + 1), latencies, color=colors[1])
    axes[1, 0].set_title("Retrieval latency per query (ms)")
    axes[1, 0].set_xlabel("Query")
    if retrieval.get("status") != "ok":
        axes[1, 0].text(0.5, 0.5, "Not run\\nuse --live-pubmed", ha="center", va="center", transform=axes[1, 0].transAxes, color="#b33a3a", fontweight="bold")

    scores = [score for item in samples for score in item.get("scores", []) if safe_float(score) is not None]
    axes[1, 1].hist(scores, bins=8, color=colors[3], edgecolor="white")
    axes[1, 1].set_title("Evidence score distribution")
    axes[1, 1].set_xlabel("Final score")
    if not scores:
        axes[1, 1].text(0.5, 0.5, "No evidence scores", ha="center", va="center", transform=axes[1, 1].transAxes, color="#b33a3a", fontweight="bold")

    metrics = retrieval.get("retrieval_metrics", {})
    metric_names = ["Precision@5", "Recall@5", "MRR"]
    metric_values = [metrics.get(name.lower().replace("@", "@"), 0) for name in metric_names]
    if not any(metric_values):
        metric_values = [0, 0, 0]
    axes[1, 2].bar(metric_names, metric_values, color=colors[4])
    axes[1, 2].set_title("Retrieval quality metrics")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].tick_params(axis="x", rotation=25)
    if metrics.get("status") == "unavailable":
        axes[1, 2].text(0.5, 0.5, "Unavailable\\nadd ground truth", ha="center", va="center", transform=axes[1, 2].transAxes, color="#b33a3a", fontweight="bold")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def markdown_report(report: dict[str, Any]) -> str:
    models = report["models"]
    return f"""# MythLens RAG Evaluation

Generated: `{report['generated_at']}`

## Executive summary

- Embedding model: `{models['embedding']['name']}`
- Reranker model: `{models['reranker']['name']}`
- Local FAISS assets: `{report['chunking']['status']}`
- Retrieval benchmark: `{report['retrieval']['retrieval_metrics'].get('status', 'measured')}`
- Dashboard: `rag_dashboard.png`

## Measurements

### Embeddings

```json
{json.dumps(report['embeddings'], indent=2)}
```

### Chunking

```json
{json.dumps(report['chunking'], indent=2)}
```

### Retrieval, similarity, and latency

```json
{json.dumps(report['retrieval'], indent=2)}
```

## Interpretation

Precision@K, Recall@K, and MRR require a labeled ground-truth file. Without committed FAISS metadata or labels, this run reports model loading, embedding geometry, chunk statistics, latency, and score distributions without inventing quality scores.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/rag_eval")
    parser.add_argument("--load-models", action="store_true", help="Load embedding and reranker models and measure them")
    parser.add_argument("--live-pubmed", action="store_true", help="Run live retrieval samples; can take time and use network")
    parser.add_argument("--ground-truth", type=Path, help="JSON labels aligned with the default queries")
    parser.add_argument("--query", action="append", dest="queries", help="Evaluation query; repeat for multiple queries")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8")) if args.ground_truth else None
    output_dir = Path(args.output_dir)
    models = model_inventory(args.load_models)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "queries": queries},
        "models": models,
        "chunking": chunking_report(),
        "embeddings": embedding_report(queries, args.load_models),
        "retrieval": retrieval_report(queries, ground_truth, args.live_pubmed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rag_evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "rag_evaluation.md").write_text(markdown_report(report), encoding="utf-8")
    try:
        plot_dashboard(report, output_dir / "rag_dashboard.png")
        report["dashboard"] = str(output_dir / "rag_dashboard.png")
        (output_dir / "rag_evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        report["dashboard_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"output_dir": str(output_dir), "dashboard": report.get("dashboard"), "chunking_status": report["chunking"]["status"], "embedding_status": report["embeddings"]["status"], "retrieval_metrics": report["retrieval"]["retrieval_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
