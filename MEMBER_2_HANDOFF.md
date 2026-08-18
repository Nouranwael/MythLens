# Member 2 handoff: MythLens Member 1 output

This folder contains the completed Member 1 work and is ready for Member 2 to build the retrieval and verification pipeline on top of it.

## What Member 1 already provides

Member 1 is responsible for:
- video URL and local file ingestion
- audio extraction and transcription
- Arabic and English language detection
- normalization for Egyptian Arabic text
- medical claim extraction
- generation of retrieval-friendly medical queries

## Main files to use

- `backend/main.py`  
  Public API entry point for the Member 1 contract.

- `backend/ingestion/transcription.py`  
  Handles downloading/extracting audio and transcription logic.

- `backend/arabic/normalization.py`  
  Detects language and normalizes Arabic text.

- `backend/claims/extraction.py`  
  Extracts medical claims and builds queries for evidence retrieval.

## Core functions

From `backend.main` you can call:

```python
from backend.main import process_text_input, process_video_input, transcribe_video
```

### 1) Text input

```python
from backend.main import process_text_input

payload = process_text_input(
    "Metformin is the first-line treatment for Type 2 Diabetes Mellitus."
)
print(payload)
```

### 2) Video URL or local file

```python
from backend.main import process_video_input

payload = process_video_input("https://example.com/video.mp4")
print(payload)
```

### 3) Direct transcription

```python
from backend.main import transcribe_video

text = transcribe_video("https://example.com/video.mp4")
print(text)
```

## Output contract for Member 2

This is the structure you should consume:

```json
{
  "original_transcript": "...",
  "language": "ar-EG",
  "summary": "...",
  "claims": [
    {
      "original_claim": "...",
      "normalized_claim": "...",
      "medical_query": "..."
    }
  ]
}
```

### Example

```json
{
  "original_transcript": "Metformin is the first-line treatment for Type 2 Diabetes Mellitus.",
  "language": "en-US",
  "summary": "Metformin is the first-line treatment for Type 2 Diabetes Mellitus.",
  "claims": [
    {
      "original_claim": "Metformin is the first-line treatment for Type 2 Diabetes Mellitus",
      "normalized_claim": "Metformin is the first-line treatment for Type 2 Diabetes Mellitus",
      "medical_query": "Metformin is the first-line treatment for Type 2 Diabetes Mellitus"
    }
  ]
}
```

## How to run the repo

From the project root:

```bash
pip install -r requirements.txt
python -c "from backend.main import process_text_input; import json; text='Metformin is the first-line treatment for Type 2 Diabetes Mellitus.'; print(json.dumps(process_text_input(text), ensure_ascii=False, indent=2))"
```

On Windows PowerShell:

```powershell
Set-Location -Path 'e:\AIHachthon\main\MythLens'
python -c "from backend.main import process_text_input; import json; text='Metformin is the first-line treatment for Type 2 Diabetes Mellitus.'; print(json.dumps(process_text_input(text), ensure_ascii=False, indent=2))"
```

## What Member 2 should build next

Your next task is to consume each `medical_query` and build the retrieval layer:

1. Retrieve trusted medical evidence
2. Search guideline data and PubMed sources
3. Rank results by relevance
4. Return evidence objects with:
   - source
   - title
   - text
   - url
   - pmid
   - study_type
   - score

Suggested target function:

```python
retrieve_evidence(medical_query, top_k=5)
```

## Important notes

- Member 1 keeps the original transcript exactly as it was extracted.
- Arabic content is normalized while preserving meaning.
- Queries are intentionally retrieval-friendly and designed for the next module.
- The production logic is in `backend/`, not in test files.

## Recommended next step

Start by building the retrieval layer in `backend/rag/` and then connect it to the `medical_query` values returned by `process_text_input`.

