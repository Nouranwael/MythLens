# Local Vector Assets

MythLens can use generated FAISS assets for the local side of its hybrid RAG pipeline. These files are intentionally not committed to GitHub because they are generated binary assets.

Place the following files in this directory before running local FAISS/BM25 retrieval:

- `medical_faiss.index` — about 40 MB
- `chunk_metadata.pkl` — about 9 MB

`medical_embeddings.npy` is not required at runtime by the integrated pipeline.

Expected layout:

```text
backend/rag/vector_store/
├── medical_faiss.index
├── chunk_metadata.pkl
└── README.md
```

Alternatively, store the assets elsewhere and set:

```env
MYTHLENS_VECTOR_PATH=/absolute/path/to/vector_store
```

If the vector assets are missing, MythLens skips local FAISS/BM25 retrieval gracefully and can still use live PubMed retrieval.
