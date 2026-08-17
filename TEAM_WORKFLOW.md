# MythLens Team Workflow

This file defines the four workstreams, file ownership, and the handoff contract between modules.

## Member 1 — Video, Transcription & Egyptian Arabic

Owns:
- `backend/ingestion/`
- `backend/arabic/`
- `backend/claims/`
- `notebooks/transcription_test.ipynb` (or equivalent test notebook)

Responsibilities:
- Video upload / URL handling
- Audio extraction
- Speech-to-text transcription
- Preserve the original Egyptian Arabic transcript
- Egyptian Arabic normalization
- Extract one or more atomic medical claims from the transcript or text input
- Generate a normalized Arabic claim and an English biomedical search query

Expected output contract:
```json
{
  "original_transcript": "...",
  "language": "ar-EG",
  "claims": [
    {
      "original_claim": "...",
      "normalized_claim": "...",
      "medical_query": "..."
    }
  ]
}
```

Main functions to expose:
```python
transcribe_video(video_input)
normalize_egyptian_arabic(text)
extract_claims(text)
generate_medical_query(claim)
```

## Member 2 — Medical RAG, Data & Retrieval

Owns:
- `backend/rag/`
- `data/guidelines/`
- `data/pubhealth/`
- `data/healthfc/`
- RAG notebooks under `notebooks/`

Responsibilities:
- Prepare WHO / CDC / FDA guideline data
- Chunk and embed trusted guideline content
- Build the local vector store
- Integrate PubMed live retrieval
- Implement hybrid retrieval
- Implement re-ranking
- Return evidence with source metadata and relevance scores

Expected input:
```json
{
  "medical_query": "topical garlic wound infection antibiotics"
}
```

Expected output contract:
```json
{
  "evidence": [
    {
      "source": "PubMed",
      "title": "...",
      "text": "...",
      "url": "...",
      "pmid": "...",
      "study_type": "Systematic Review",
      "score": 0.91
    }
  ]
}
```

Main function to expose:
```python
retrieve_evidence(medical_query, top_k=5)
```

## Member 3 — Verification, Clinical Safety & Evaluation

Owns:
- `backend/verification/`
- `backend/evaluation/`
- `data/egyptian_myths/`
- Evaluation notebooks under `notebooks/`

Responsibilities:
- Determine stance: Supported / Refuted / Misleading / Unproven
- Evidence synthesis grounded only in retrieved sources
- Clinical risk assessment
- Safety recommendations
- Build the Egyptian Arabic myth evaluation set
- Evaluate with PUBHEALTH and HealthFC
- Implement classification and retrieval metrics

Expected input:
```json
{
  "claim": {
    "original_claim": "...",
    "normalized_claim": "..."
  },
  "evidence": []
}
```

Expected output contract:
```json
{
  "verdict": "MISLEADING",
  "risk_level": "HIGH",
  "confidence": 0.91,
  "explanation_ar": "...",
  "safe_recommendation": "...",
  "insufficient_evidence": false,
  "citations": []
}
```

Main functions to expose:
```python
verify_claim(claim, evidence)
assess_risk(claim, evidence)
evaluate_verdicts(...)
evaluate_retrieval(...)
```

Key metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Recall@K
- Precision@K
- MRR

## Member 4 — Frontend, API & Final Integration

Owns:
- `frontend/`
- `backend/main.py`
- Integration code connecting all modules
- Final end-to-end demo

Responsibilities:
- Build the MythLens interface
- Support two input modes: video and text claim
- Connect the full backend pipeline
- Display transcript, extracted claims, verdict, risk level, explanation, and citations
- Handle loading and error states
- Run final end-to-end integration testing

Final pipeline:
```python
transcript = transcribe_video(video_input)
claims = extract_claims(transcript)

results = []
for claim in claims:
    evidence = retrieve_evidence(claim["medical_query"])
    result = verify_claim(claim, evidence)
    results.append(result)
```

## Rules for Everyone

1. Work only inside your assigned folders unless the team agrees otherwise.
2. Do not rename shared function names or JSON fields without telling the team.
3. Keep experimental work in `notebooks/`, but move final reusable code into `backend/`.
4. Never commit API keys, tokens, passwords, or `.env` files.
5. Large datasets should not be committed to GitHub. Keep them in Google Drive or download them through scripts.
6. Before starting work, pull the latest code.
7. Commit small, clear changes with descriptive messages.
8. The `main` branch is the stable shared version.

## Suggested Work Order

### Phase 1 — Independent modules
- Member 1: transcription + Arabic + claim extraction
- Member 2: guideline ingestion + PubMed + retrieval
- Member 3: verifier + safety + evaluation setup
- Member 4: UI skeleton + API skeleton

### Phase 2 — First integration
- Connect text claim → retrieval → verification first
- Test the system without video

### Phase 3 — Video integration
- Connect transcription output to the working text-claim pipeline

### Phase 4 — Evaluation & demo hardening
- Run benchmark metrics
- Test Egyptian Arabic cases
- Verify citations and safety behavior
- Prepare one stable live-demo case and one backup case
