"""Evidence-grounded medical claim verification for MythLens."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

ALLOWED_VERDICTS = {"SUPPORTED", "REFUTED", "MISLEADING", "UNPROVEN"}


def _claim_text(claim: Any) -> str:
    if isinstance(claim, dict):
        return str(claim.get("normalized_claim") or claim.get("original_claim") or "").strip()
    return str(claim or "").strip()


def _citation_objects(evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"source": str(e.get("source", "")), "title": str(e.get("title", "")), "url": str(e.get("url", "")), "pmid": str(e.get("pmid", ""))} for e in evidence[:5]]


def _call_llm_json(prompt: str):
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_VERIFIER_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a conservative medical fact-checking verifier. Use ONLY supplied evidence. If evidence is insufficient or conflicting choose UNPROVEN. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None


def verify_claim(claim: Any, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = _claim_text(claim)
    evidence = evidence or []
    citations = _citation_objects(evidence)
    if not evidence:
        return {"verdict": "UNPROVEN", "confidence": 0.0, "explanation_ar": "مفيش أدلة طبية كافية متاحة في نتائج البحث عشان نأكد أو ننفي الادعاء ده.", "citations": [], "insufficient_evidence": True}

    evidence_text = "\n\n".join(
        f"[{i+1}] Source: {e.get('source','')}\nTitle: {e.get('title','')}\nStudy type: {e.get('study_type','')}\nEvidence: {e.get('text','')}\nURL: {e.get('url','')}"
        for i, e in enumerate(evidence[:5])
    )
    prompt = f"""Medical claim:\n{text}\n\nRetrieved evidence:\n{evidence_text}\n\nChoose exactly one verdict: SUPPORTED, REFUTED, MISLEADING, or UNPROVEN. Return JSON with verdict, confidence from 0 to 1, explanation_ar, insufficient_evidence."""
    result = _call_llm_json(prompt)
    if not result:
        return {"verdict": "UNPROVEN", "confidence": 0.0, "explanation_ar": "الأدلة اتجمعت، لكن أداة التحقق مش متاحة دلوقتي؛ عشان كده مش هنفترض حكم طبي من غير تحقق.", "citations": citations, "insufficient_evidence": True}

    verdict = str(result.get("verdict", "UNPROVEN")).upper().strip()
    if verdict not in ALLOWED_VERDICTS:
        verdict = "UNPROVEN"
    try:
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    insufficient = bool(result.get("insufficient_evidence", verdict == "UNPROVEN")) or verdict == "UNPROVEN"
    return {"verdict": verdict, "confidence": confidence, "explanation_ar": str(result.get("explanation_ar") or "الأدلة الحالية غير كافية لإصدار حكم موثوق."), "citations": citations, "insufficient_evidence": insufficient}
