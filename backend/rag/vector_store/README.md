# Member 2 Vector Assets

The local hybrid RAG uses Member 2's generated FAISS assets. These files are intentionally not committed to GitHub because they are large generated binaries.

Copy these two required files from Member 2's handoff into this directory before running local FAISS/BM25 retrieval:

- `medical_faiss.index` — about 40 MB
- `chunk_metadata.pkl` — about 9 MB

`medical_embeddings.npy` is not required at runtime by the integrated pipeline and does not need to be copied.

Expected layout:

```text
backend/rag/vector_store/
├── medical_faiss.index
├── chunk_metadata.pkl
└── README.md
```

Alternatively, put the assets somewhere else and set:

```env
MYTHLENS_VECTOR_PATH=/absolute/path/to/vector_store
```

If the vector assets are missing, MythLens skips local FAISS/BM25 retrieval gracefully and can still use live PubMed retrieval.
