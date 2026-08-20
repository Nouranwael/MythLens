"""Final integration entry point for the MythLens backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from backend.arabic.normalization import detect_language, normalize_egyptian_arabic
from backend.claims.extraction import extract_claims, generate_medical_query
from backend.ingestion.transcription import process_text_input, process_video_input, summarize_video_transcript, transcribe_video


def prepare_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    claims = payload.get("claims", []) or []
    return {
        "original_transcript": payload.get("original_transcript", ""),
        "language": payload.get("language", ""),
        "summary": payload.get("summary", ""),
        "claims_count": len(claims),
        "claims": claims,
        "results": [],
        "verification_status": "pending",
    }


def _verify_one_claim(claim: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    """Retrieve and verify one claim independently for concurrent execution."""
    from backend.verification import assess_risk, verify_claim

    retrieval_error = None
    evidence: List[Dict[str, Any]] = []
    try:
        from backend.rag import retrieve_evidence
        retrieval_payload = retrieve_evidence(claim.get("medical_query", ""), top_k=top_k)
        evidence = retrieval_payload.get("evidence", []) or []
    except Exception as exc:
        retrieval_error = str(exc)

    verification = verify_claim(claim, evidence)
    safety = assess_risk(claim, evidence)
    return {
        **claim,
        "verdict": verification.get("verdict", "UNPROVEN"),
        "confidence": verification.get("confidence", 0.0),
        "risk_level": safety.get("risk_level", "LOW"),
        "explanation_ar": verification.get("explanation_ar", ""),
        "safe_recommendation": safety.get("safe_recommendation", ""),
        "detected_risks": safety.get("detected_risks", []),
        "insufficient_evidence": verification.get("insufficient_evidence", True),
        "citations": verification.get("citations", []),
        "evidence": evidence,
        "retrieval_error": retrieval_error,
    }


def verify_payload(payload: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    claims = payload.get("claims", []) or []
    results: List[Dict[str, Any] | None] = [None] * len(claims)
    max_workers = min(4, max(1, len(claims)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_verify_one_claim, claim, top_k): index
            for index, claim in enumerate(claims)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return {
        **prepare_payload(payload),
        "results": [result for result in results if result is not None],
        "verification_status": "complete",
    }


def _fact_check_payload(payload: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    return verify_payload(payload, top_k=top_k)


def analyze_text_claim(text: str, top_k: int = 5) -> Dict[str, Any]:
    return _fact_check_payload(process_text_input(text), top_k=top_k)


def analyze_video_input(video_input: Any, top_k: int = 5) -> Dict[str, Any]:
    return _fact_check_payload(process_video_input(video_input), top_k=top_k)


def main() -> None:
    print("MythLens backend is ready. Use analyze_text_claim() or analyze_video_input().")


if __name__ == "__main__":
    main()
