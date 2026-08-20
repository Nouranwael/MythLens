"""Medical claim extraction and retrieval-query generation."""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv

from backend.llm.client import chat_text

try:
    from groq import Groq
except ImportError:
    Groq = None

load_dotenv()

MEDICAL_KEYWORDS = [
    "treatment", "cure", "healing", "medicine", "medication", "symptom", "infection", "diagnosis",
    "doctor", "hospital", "viral", "bacterial", "pain", "fever", "cough", "remedy", "wound",
    "allergy", "pregnant", "metformin", "diabetes", "insulin", "gluconeogenesis", "hypoglycemia",
    "type 2", "glucose", "glycemic", "therapy", "clinical", "drug", "pharmacological", "alcohol",
    "immune", "virus", "milk", "bone", "bones", "wine", "cold weather", "strong bones",
    "broken bones", "good for you", "make you sick", "immune system", "disease", "health",
    "nutrition", "vitamin", "sick", "probiotic", "water", "hydration", "hydrated", "fluid",
    "eight glasses", "daily intake", "vegetables", "fruits", "exercise", "dehydration", "thirst",
    "cancer", "radiation", "x-ray", "xray", "microwave", "coffee", "breakfast", "growth",
    "anxiety", "sleep", "electromagnetic", "ionizing", "non-ionizing", "health myth", "myth",
    "risk", "benefit", "danger", "علاج", "دواء", "أعراض", "عدوى", "تشخيص", "طبيب", "سعال",
    "حمى", "ألم", "شفاء", "جرح", "حساسية", "حمل", "سكري", "غلوكوز", "أنسولين", "ميتفورمين",
    "كحول", "مناعة", "فيروس", "حليب", "عظام", "نبيذ", "مريض", "صحة", "تغذية", "ماء",
    "ترطيب", "سائل", "ثمانية أكواب", "نباتات", "فاكهة", "تمارين", "جفاف", "سرطان",
    "إشعاع", "قهوة", "فطور", "أرق", "قلق", "ثوم", "ليمون", "صيام"
]

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")


class GroqExtractionError(RuntimeError):
    """Raised when the required Groq claim-extraction step is unavailable."""


class GroqSummaryError(RuntimeError):
    """Raised when the required Groq transcript summary step is unavailable."""

ARABIC_MEDICAL_GLOSSARY = {
    "الثوم": "garlic", "ثوم": "garlic", "الليمون": "lemon", "ليمون": "lemon",
    "العسل": "honey", "عسل": "honey", "الصيام": "fasting", "صيام": "fasting",
    "المعدة": "stomach", "معدة": "stomach", "القولون": "colon irritable bowel",
    "قولون": "colon irritable bowel", "السكري": "diabetes mellitus", "سكري": "diabetes mellitus",
    "السكر": "blood glucose diabetes", "أنسولين": "insulin", "الأنسولين": "insulin",
    "الضغط": "blood pressure hypertension", "ضغط": "blood pressure hypertension",
    "السرطان": "cancer", "سرطان": "cancer", "المناعة": "immune system", "مناعة": "immune system",
    "جرح": "wound", "الجروح": "wounds", "التهاب": "inflammation", "حرق": "burn", "حروق": "burns",
    "مضاد حيوي": "antibiotic", "المضاد الحيوي": "antibiotic", "دواء": "medication",
    "علاج": "treatment", "يعالج": "treats", "يشفي": "cures", "يمنع": "prevents",
    "يسبب": "causes", "مفيد": "benefit", "مضر": "harm risk", "خطر": "risk",
    "فيروس": "virus", "عدوى": "infection", "بكتيريا": "bacteria", "فيتامين": "vitamin",
    "مياه": "water hydration", "ماء": "water hydration", "قهوة": "coffee", "النوم": "sleep",
    "نوم": "sleep", "قلق": "anxiety"
}


def _contains_medical_context(text: str) -> bool:
    lower = str(text or "").lower()
    return any(keyword.lower() in lower for keyword in MEDICAL_KEYWORDS)


def _strip_non_important_symbols(text: str) -> str:
    cleaned = str(text or "")
    for symbol in ("**", "__", "*", "_", "`", "•", "–", "—", "[", "]", "(", ")", "{", "}", "<", ">", "#"):
        cleaned = cleaned.replace(symbol, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;!?/\\|\t\n")
    return cleaned


def _get_groq_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or Groq is None:
        return None
    return Groq(api_key=api_key)


def _parse_groq_claim_payload(content: str):
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"```(?:json)?", "", content, flags=re.IGNORECASE).replace("```", "").strip()
    decoder = json.JSONDecoder()
    payloads = []
    for match in re.finditer(r"\{", content):
        try:
            candidate, _ = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("claims"), list):
            payloads.append(candidate)
    return max(payloads, key=lambda item: len(item.get("claims", [])), default=None)


def _extract_claims_with_groq(text: str):
    """Extract claims with Groq; do not silently replace the LLM with heuristics."""
    client = _get_groq_client()
    if client is None:
        raise GroqExtractionError("Set GROQ_API_KEY and GROQ_MODEL in .env.")

    system_prompt = """You extract health-related claims from transcripts in Arabic, English, or mixed language.
Return only one valid JSON object with this shape: {\"claims\": [{\"original_claim\": \"...\", \"normalized_claim\": \"...\", \"medical_query\": \"...\"}]}.
Rules:
- Use only information explicitly stated in the transcript. Do not add facts or correct the speaker.
- Include medical claims and general health advice, myths, risks, benefits, symptoms, prevention, nutrition, medication, and treatment claims.
- Exclude greetings, opinions without a health assertion, advertisements, narration, and filler.
- Split separate assertions into atomic claims. Keep enough context for each claim to stand alone.
- Preserve Arabic claims in Arabic and English claims in English in original_claim and normalized_claim.
- Keep normalized_claim concise while preserving the speaker's meaning.
- Write medical_query in concise English suitable for PubMed or guideline search. Translate Arabic claims when needed.
- Do not include reasoning, markdown, or any text outside the JSON object.
- If there are no health-related claims, return {\"claims\": []}."""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _strip_non_important_symbols(text)},
        ]
        for attempt in range(2):
            response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", GROQ_MODEL),
                temperature=0.1,
                reasoning_format="hidden",
                max_completion_tokens=4096,
                messages=messages,
            )
            payload = _parse_groq_claim_payload(response.choices[0].message.content or "")
            if payload is not None:
                break
            messages[0] = {
                "role": "system",
                "content": "Return only valid JSON: {\"claims\":[{\"original_claim\":\"...\",\"normalized_claim\":\"...\",\"medical_query\":\"...\"}]}. Extract separate Arabic or English health claims from the transcript. No reasoning or markdown.",
            }
        else:
            raise GroqExtractionError("Groq returned no valid claims JSON after retry.")
    except GroqExtractionError:
        raise
    except Exception as exc:
        raise GroqExtractionError(f"Groq claim extraction failed: {exc}") from exc

    claims = []
    seen = set()
    for item in payload.get("claims", []):
        if not isinstance(item, dict):
            continue
        original = re.sub(r"\s+", " ", str(item.get("original_claim", "")).strip())
        normalized = re.sub(r"\s+", " ", str(item.get("normalized_claim", original)).strip())
        query = re.sub(r"\s+", " ", str(item.get("medical_query", "")).strip())
        if not original or not normalized or not query:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        claims.append({
            "original_claim": original.rstrip(". "),
            "normalized_claim": normalized.rstrip(". "),
            "medical_query": query.rstrip(". "),
        })
    return claims


def summarize_with_groq(text: str, language: str = "", claims: list[dict] | None = None) -> str:
    """Summarize a cleaned transcript in its detected language using Groq."""
    client = _get_groq_client()
    if client is None:
        raise GroqSummaryError("Set GROQ_API_KEY and GROQ_MODEL in .env.")

    output_language = "Egyptian Arabic" if language == "ar-EG" else "English"
    prompt = (
        f"Summarize this health video transcript in clear {output_language}. "
        "Write 2 or 3 natural sentences covering only the main health information. "
        "Do not add facts, commentary, headings, or a conclusion. Return only the summary."
    )
    claim_context = ""
    if claims:
        claim_context = "\n\nUse these extracted claims to understand unclear speech, but do not add information:\n" + "\n".join(
            f"- {claim.get('normalized_claim', '')}" for claim in claims[:10]
        )
    try:
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", GROQ_MODEL),
            temperature=0.1,
            reasoning_format="raw",
            max_completion_tokens=2048,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": _strip_non_important_symbols(text) + claim_context},
            ],
        )
        summary = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.DOTALL | re.IGNORECASE)
        summary = re.sub(r"\s+", " ", summary).strip(" \"'")
        if not summary:
            raise GroqSummaryError("Groq returned an empty summary.")
        return summary
    except GroqSummaryError:
        raise
    except Exception as exc:
        raise GroqSummaryError(f"Groq summary generation failed: {exc}") from exc


def _split_sentences(text: str):
    text = _strip_non_important_symbols(text)
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|(?<=[،])\s*", text) if part.strip()]


def _split_atomic_clauses(sentence: str):
    text = _strip_non_important_symbols(sentence)
    if not text:
        return []
    if ARABIC_RANGE.search(text):
        chunks = re.split(r"\s+(?:و|لكن|حيث|كما|ولا)\s+", text)
    else:
        chunks = re.split(r"(?<=[.!?])\s+|(?<=[,;])\s+|\s+(?:and|or|but|which|where|while|however|therefore|thus|since|because|although)\s+", text, flags=re.IGNORECASE)
    return [chunk.strip().rstrip(". ") for chunk in chunks if len(chunk.strip().split()) >= 3 and _contains_medical_context(chunk)]


def _required_glossary_terms(arabic_claim: str):
    required = []
    for ar, en in sorted(ARABIC_MEDICAL_GLOSSARY.items(), key=lambda pair: len(pair[0]), reverse=True):
        if ar in arabic_claim:
            head = en.split()[0].lower()
            if head not in required:
                required.append(head)
    return required


def _llm_biomedical_query(claim: str):
    is_arabic = bool(ARABIC_RANGE.search(claim))
    required_terms = _required_glossary_terms(claim) if is_arabic else []
    glossary_hint = ", ".join(required_terms) if required_terms else "none"

    query = chat_text(
        "Convert the medical claim into a concise English PubMed search query. Use key biomedical concepts and useful MeSH-style Boolean terms when helpful. Preserve intervention, condition, claimed effect, population, dose and duration when present. Output ONLY the query, with no explanation. If the claim is Arabic, translate medical concepts accurately. Never translate ثوم as thymus; ثوم means garlic.",
        f"Claim: {claim}\nRequired biomedical terms from glossary: {glossary_hint}",
        purpose="query",
    )
    if not query:
        return None

    query = query.strip().strip('"')
    if not query or ARABIC_RANGE.search(query):
        return None

    if "garlic" in required_terms:
        query = re.sub(r"\bthymus\b", "garlic", query, flags=re.IGNORECASE)

    lower_query = query.lower()
    missing = [term for term in required_terms if term not in lower_query]
    if missing:
        query = f"{query} {' '.join(missing)}"

    return re.sub(r"\s+", " ", query)


def _fallback_arabic_query(arabic_claim: str) -> str:
    matched = []
    for ar, en in sorted(ARABIC_MEDICAL_GLOSSARY.items(), key=lambda pair: len(pair[0]), reverse=True):
        if ar in arabic_claim and en not in matched:
            matched.append(en)
    return " ".join(matched + ["clinical evidence safety"]) if matched else "medical claim clinical evidence safety"


def generate_medical_query(claim: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(claim or "").strip()).rstrip(". ")
    if not cleaned:
        return "clinical evidence for medical claim"

    llm_query = _llm_biomedical_query(cleaned)
    if llm_query:
        return llm_query

    if ARABIC_RANGE.search(cleaned):
        return _fallback_arabic_query(cleaned)

    return f"clinical evidence {cleaned}"


def extract_claims(text: str):
    if text is None:
        return []

    cleaned_text = _strip_non_important_symbols(text)
    if not cleaned_text:
        return []

    return _extract_claims_with_groq(cleaned_text)
