# MythLens

AI-powered medical myth fact-checking system using RAG, PubMed evidence, and Egyptian Arabic support.

## Core Flow

Video URL / Upload or Text Claim -> Transcription (for video) -> Egyptian Arabic Processing -> Claim Extraction -> Medical Query Generation -> Hybrid Retrieval -> Re-ranking -> Evidence Verification -> Clinical Safety -> Structured Result.

## Repository Structure

- `backend/ingestion/` - video loading, audio extraction, and transcription.
- `backend/arabic/` - Egyptian Arabic normalization and medical-language processing.
- `backend/claims/` - claim extraction and medical query generation.
- `backend/rag/` - vector search, PubMed retrieval, hybrid retrieval, and re-ranking.
- `backend/verification/` - stance verification and clinical safety logic.
- `backend/evaluation/` - retrieval and verdict evaluation metrics.
- `frontend/` - user interface and final app integration.
- `data/` - local development datasets and guideline placeholders.
- `notebooks/` - Colab experiments before stable code is moved into `backend/`.

## Team Workflow

Each teammate can prototype their module in Google Colab. Once a component is stable, move the reusable functions into the matching folder under `backend/` and push the changes to GitHub. GitHub is the shared source of truth; VS Code can be used for final integration and end-to-end testing.

## Important Rule

Do not commit API keys, secrets, large raw datasets, downloaded model weights, or generated vector databases. Use `.env` locally and keep only `.env.example` in the repository.
