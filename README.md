# MythLens

MythLens is an AI-powered medical myth fact-checking system designed for Arabic, Egyptian Arabic, and English health content. It accepts a direct text claim, a video URL, or an uploaded video/audio file, extracts medical claims, retrieves scientific evidence, and returns a structured verdict with safety guidance and citations.

## Product Demo

### Input Interface

<p align="center">
  <img src="docs/images/mythlens-home.png" alt="MythLens input interface" width="92%" />
</p>

### Fact-Check Results

<table>
  <tr>
    <td width="50%" align="center"><strong>Evidence-limited / Unproven Example</strong></td>
    <td width="50%" align="center"><strong>Supported Example</strong></td>
  </tr>
  <tr>
    <td><img src="docs/images/mythlens-result-unproven.png" alt="MythLens unproven fact-check result" /></td>
    <td><img src="docs/images/mythlens-result-supported.png" alt="MythLens supported fact-check result" /></td>
  </tr>
</table>

The interface surfaces claim-level verdicts, confidence, evidence sufficiency, clinical risk, safe recommendations, and linked PubMed evidence with PMID identifiers.

## System Architecture

<p align="center">
  <img src="docs/images/mythlens-architecture.svg" alt="MythLens system architecture" width="100%" />
</p>

The pipeline separates evidence retrieval, verdict generation, and clinical safety so that a claim can be evaluated for scientific support and medical risk independently.

## Key Features

- Text, video URL, and video/audio upload input
- Faster-Whisper transcription for media
- Arabic and Egyptian Arabic processing
- Groq-based health claim extraction and summarization
- PubMed-focused medical query generation
- Hybrid retrieval with local FAISS, BM25, and live PubMed
- Evidence re-ranking by relevance and study quality
- Evidence-grounded verification using Gemini
- Verdicts: `SUPPORTED`, `REFUTED`, `MISLEADING`, `UNPROVEN`
- Independent clinical safety assessment: `LOW`, `MODERATE`, `HIGH`, `CRITICAL`
- PubMed citations with PMID and source links
- FastAPI backend with a lightweight web interface
- Reproducible retrieval evaluation utilities

## System Flow

```text
Text / Video URL / Video Upload
        ↓
Transcription (media only)
        ↓
Arabic / Egyptian Arabic processing
        ↓
Claim extraction + transcript summary
        ↓
Medical query generation
        ↓
Hybrid RAG: FAISS + BM25 + PubMed
        ↓
Cross-encoder re-ranking
        ↓
Evidence-grounded verification
        ↓
Clinical safety assessment
        ↓
Structured fact-check result + citations
```

## Final RAG V2 Configuration

After comparing multiple embedding, chunking, and reranking configurations, the final retrieval setup is:

| Component | Final Configuration |
| --- | --- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Chunk size | `180 words` |
| Chunk overlap | `30 words` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Final Top-K | `5` |
| Local candidate pool | `20` |
| PubMed candidate pool | `20` |
| FAISS candidate pool | `50` |

The original local embedding model was `pritamdeka/S-PubMedBert-MS-MARCO`. RAG V2 adopts MiniLM because it produced the strongest overall retrieval performance on the final judged benchmark while remaining lightweight and fast.

## Tech Stack

- Python 3.11+
- FastAPI + Uvicorn
- Faster-Whisper
- Groq (`qwen/qwen3.6-27b` by default)
- Gemini Interactions API
- PubMed / NCBI E-utilities
- FAISS
- Sentence Transformers
- BM25
- Cross-encoder re-ranking
- Vanilla HTML, CSS, and JavaScript

## Repository Structure

```text
MythLens/
├── backend/
│   ├── ingestion/        # video/audio ingestion and transcription
│   ├── arabic/           # Arabic/Egyptian Arabic processing
│   ├── claims/           # health claim extraction and medical queries
│   ├── llm/              # Gemini REST client
│   ├── rag/              # vector search, BM25, PubMed and re-ranking
│   ├── verification/     # verdict generation and clinical safety
│   ├── evaluation/       # evaluation metrics
│   ├── api.py            # FastAPI application
│   └── main.py           # end-to-end integration
├── frontend/             # web interface
├── docs/
│   └── images/           # screenshots, architecture, evaluation dashboard
├── data/                 # local/evaluation data (large files ignored by Git)
├── scripts/
│   ├── build_rag_v2.py          # build the final local FAISS store
│   ├── evaluate_rag.py          # baseline evaluation utility
│   └── evaluate_hybrid_rag.py   # manually judged Hybrid RAG evaluation
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nouranwael/MythLens.git
cd MythLens
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env`, then add your keys:

```env
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-3.5-flash-lite

GROQ_API_KEY=your_groq_key
GROQ_MODEL=qwen/qwen3.6-27b

NCBI_API_KEY=
NCBI_EMAIL=
MYTHLENS_LLM_DEBUG=false
```

`GROQ_API_KEY` is required for the current claim-extraction and transcript-summary pipeline. `GEMINI_API_KEY` is required for evidence verification and Gemini-backed query generation where used. NCBI credentials are optional but recommended for higher PubMed request limits.

Never commit `.env` or real API keys.

## Build the Final Local RAG Store

The large source datasets and generated FAISS files are intentionally not committed to GitHub. With the required local datasets placed under `data/`, build the final RAG V2 store with:

```bash
python scripts/build_rag_v2.py --model sentence-transformers/all-MiniLM-L6-v2 --chunk-size 180 --overlap 30 --output-dir backend/rag/vector_store
```

The application defaults to this final MiniLM configuration through `backend/rag/config.py`.

## Run the Web App

```bash
uvicorn backend.api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

If port 8000 is already in use:

```bash
uvicorn backend.api:app --reload --port 8001
```

## Main API Endpoints

- `GET /api/health` — service/configuration health check
- `POST /api/analyze/text` — full text fact-check pipeline
- `POST /api/analyze/url` — full video-URL pipeline
- `POST /api/analyze/video` — full uploaded media pipeline
- `POST /api/prepare/text` — claim extraction without verification
- `POST /api/prepare/url` — URL transcription + claim extraction
- `POST /api/prepare/video` — uploaded media transcription + claim extraction
- `POST /api/verify` — verify an already prepared payload

Interactive API documentation is available at `/docs` while the server is running.

## Final Retrieval Evaluation

MythLens RAG V2 was evaluated on a manually judged 12-query medical retrieval benchmark. Each query produced a 10-candidate evidence pool and every candidate was assigned one of three relevance grades:

- `0` — irrelevant
- `1` — partially relevant / useful supporting context
- `2` — directly relevant medical evidence

Standard metrics treat grades `1` and `2` as relevant. Strict metrics count only grade `2` as relevant.

| Metric | Standard | Strict |
| --- | ---: | ---: |
| Precision@5 | **88.33%** | **56.67%** |
| Recall@5 | **61.48%** | **82.40%*** |
| MRR | **100.00%** | **68.33%** |
| nDCG@5 | **86.32%** | **71.24%** |
| Retrieval Coverage | **100.00%** | — |
| Direct Evidence Coverage@5 | — | **91.67%** |

\* Strict Recall is averaged over the 11 queries whose judged candidate pool contained at least one directly relevant (`grade 2`) result.

<p align="center">
  <img src="docs/images/hybrid_rag_dashboard.png" alt="MythLens Hybrid RAG evaluation dashboard" width="96%" />
</p>

### RAG V2 Improvement over the Previous Embedding Setup

The same evaluation methodology was used to compare the previous retrieval setup with the final MiniLM RAG V2 configuration.

| Metric | Previous Setup | Final RAG V2 |
| --- | ---: | ---: |
| Precision@5 | 83.33% | **88.33%** |
| Recall@5 | 59.17% | **61.48%** |
| MRR | 100.00% | **100.00%** |
| nDCG@5 | 77.99% | **86.32%** |
| Retrieval Coverage | 100.00% | **100.00%** |
| Strict Precision@5 | 46.67% | **56.67%** |
| Strict Recall@5 | 62.51% | **82.40%** |
| Strict MRR | 65.28% | **68.33%** |
| Strict nDCG@5 | 60.08% | **71.24%** |
| Direct Evidence Coverage@5 | 83.33% | **91.67%** |

The final RAG V2 therefore improves both top-five relevance and ranking quality while preserving complete retrieval coverage on the benchmark.

The reported Recall@5 values are calculated over the manually judged candidate pool, not over the entire PubMed/local corpus. Retrieval evaluation is separate from final verdict correctness, which requires a labeled claim-verdict test set.

### Reproduce the Hybrid RAG Evaluation

Prepare candidate evidence for manual relevance review:

```bash
python scripts/evaluate_hybrid_rag.py --prepare
```

After assigning `relevance` values (`0`, `1`, or `2`) in `outputs/hybrid_eval/review.json`, calculate both standard and strict metrics and regenerate the dashboard:

```bash
python scripts/evaluate_hybrid_rag.py --score outputs/hybrid_eval/review.json
```

Generated evaluation outputs, model files, vector indexes, experiment stores, and large datasets are intentionally excluded from Git.

## Medical Safety

MythLens is a fact-checking and decision-support prototype, not a diagnostic system. Medical outputs are grounded in retrieved evidence where available and include a separate safety-risk assessment. Users should not stop, replace, or change prescribed treatment based only on the application output.

## Security and Repository Hygiene

- `.env`, credentials, local virtual environments, caches, model files, generated vector assets, experiment outputs, and large datasets are ignored by Git.
- Only `.env.example` is committed for configuration guidance.
- No API keys are required to be stored in source files.
