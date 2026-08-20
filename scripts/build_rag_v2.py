"""Build a configurable MythLens local vector store for RAG experiments.

The builder reproduces the original local evidence preparation strategy while
making the embedding model and chunking parameters configurable.

Default experiment:
    embedding model: sentence-transformers/all-MiniLM-L6-v2
    chunk size:      180 words
    chunk overlap:   30 words

Expected local datasets:
    data/PUBHEALTH/train.tsv
    data/healthFC_annotated.csv

Generated binary assets are written to backend/rag/vector_store/ and remain
ignored by Git.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBHEALTH = ROOT / "data" / "PUBHEALTH" / "train.tsv"
DEFAULT_HEALTHFC = ROOT / "data" / "healthFC_annotated.csv"
DEFAULT_OUTPUT_DIR = ROOT / "backend" / "rag" / "vector_store"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).split()).strip()


def chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = clean_text(text).split()
    if not words:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        part = words[start : start + chunk_size]
        if not part:
            break
        chunks.append(" ".join(part))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_pubhealth(path: Path, chunk_size: int, overlap: int) -> list[dict]:
    df = pd.read_csv(path, sep="\t")
    records = []
    for row_index, row in df.iterrows():
        evidence_text = clean_text(row.get("explanation"))
        if not evidence_text:
            continue

        parent_id = f"pubhealth_train_{row_index}"
        claim = clean_text(row.get("claim"))
        label = clean_text(row.get("label"))
        for chunk_index, chunk in enumerate(chunk_words(evidence_text, chunk_size, overlap)):
            records.append(
                {
                    "id": f"{parent_id}_chunk_{chunk_index}",
                    "dataset": "PUBHEALTH",
                    "claim": claim,
                    "text": chunk,
                    "label": label,
                    "split": "train",
                    "source": "PUBHEALTH",
                    "title": f"PUBHEALTH Evidence {row_index}",
                    "url": "",
                    "pmid": "",
                    "study_type": "",
                    "parent_id": parent_id,
                    "chunk_index": chunk_index,
                }
            )
    return records


def load_healthfc(path: Path, chunk_size: int, overlap: int) -> list[dict]:
    df = pd.read_csv(path)
    records = []
    for row_index, row in df.iterrows():
        # The original MythLens vector store used the English study/evidence field
        # rather than the entire article body.
        evidence_text = clean_text(row.get("en_studies"))
        if not evidence_text:
            continue

        parent_id = f"healthfc_{row_index}"
        claim = clean_text(row.get("en_claim"))
        label = clean_text(row.get("label"))
        url = clean_text(row.get("url"))
        for chunk_index, chunk in enumerate(chunk_words(evidence_text, chunk_size, overlap)):
            records.append(
                {
                    "id": f"{parent_id}_chunk_{chunk_index}",
                    "dataset": "HealthFC",
                    "claim": claim,
                    "text": chunk,
                    "label": label,
                    "split": "retrieval",
                    "source": "HealthFC",
                    "title": f"HealthFC Evidence {row_index}",
                    "url": url,
                    "pmid": "",
                    "study_type": "",
                    "parent_id": parent_id,
                    "chunk_index": chunk_index,
                }
            )
    return records


def build_index(records: list[dict], model_name: str, output_dir: Path, batch_size: int) -> dict:
    if not records:
        raise ValueError("No evidence records were created. Check the local dataset paths.")

    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model: {model_name}")
    print(f"Device: {device}")
    model = SentenceTransformer(model_name, device=device)

    texts = [record["text"] for record in records]
    print(f"Encoding {len(texts):,} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "medical_faiss.index"
    metadata_path = output_dir / "chunk_metadata.pkl"
    manifest_path = output_dir / "rag_v2_manifest.json"

    faiss.write_index(index, str(index_path))
    with metadata_path.open("wb") as handle:
        pickle.dump(records, handle)

    manifest = {
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "chunks": len(records),
        "faiss_index": "IndexFlatIP",
        "normalized_embeddings": True,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nRAG V2 vector store built successfully")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pubhealth", type=Path, default=DEFAULT_PUBHEALTH)
    parser.add_argument("--healthfc", type=Path, default=DEFAULT_HEALTHFC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=180, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=30, help="Chunk overlap in words")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    for path in (args.pubhealth, args.healthfc):
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Place the raw PUBHEALTH and HealthFC files under data/ before rebuilding the index."
            )

    print("Preparing PUBHEALTH evidence...")
    records = load_pubhealth(args.pubhealth, args.chunk_size, args.overlap)
    pubhealth_count = len(records)

    print("Preparing HealthFC evidence...")
    healthfc_records = load_healthfc(args.healthfc, args.chunk_size, args.overlap)
    records.extend(healthfc_records)

    print(f"Chunk size: {args.chunk_size} words")
    print(f"Overlap: {args.overlap} words")
    print(f"PUBHEALTH chunks: {pubhealth_count:,}")
    print(f"HealthFC chunks: {len(healthfc_records):,}")
    print(f"Total chunks: {len(records):,}")

    manifest = build_index(records, args.model, args.output_dir, args.batch_size)
    manifest.update(
        {
            "chunk_size_words": args.chunk_size,
            "chunk_overlap_words": args.overlap,
            "pubhealth_chunks": pubhealth_count,
            "healthfc_chunks": len(healthfc_records),
        }
    )
    (args.output_dir / "rag_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
