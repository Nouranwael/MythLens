import json
import os
import re

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from groq import Groq
except ImportError:
    Groq = None

MEDICAL_KEYWORDS = [
    "treatment", "cure", "healing", "medicine", "medication", "symptom",
    "infection", "diagnosis", "doctor", "hospital", "viral", "bacterial",
    "pain", "fever", "cough", "remedy", "wound", "allergy", "pregnant",
    "metformin", "diabetes", "insulin", "gluconeogenesis", "hypoglycemia",
    "type 2", "glucose", "glycemic", "therapy", "clinical", "drug", "pharmacological",
    "alcohol", "immune", "virus", "milk", "bone", "bones", "wine", "cold weather",
    "strong bones", "broken bones", "good for you", "make you sick", "immune system",
    "disease", "health", "nutrition", "vitamin", "sick", "infection", "probiotic",
    "water", "hydration", "hydrated", "fluid", "eight glasses", "daily intake",
    "vegetables", "fruits", "exercise", "body guide", "dehydration", "thirst",
    "cancer", "radiation", "x-ray", "xray", "microwave", "coffee", "breakfast",
    "growth", "anxiety", "sleep", "electromagnetic", "ionizing", "non-ionizing",
    "health myth", "myth", "health myths", "risk", "benefit", "danger",
    "علاج", "دواء", "أعراض", "عدوى", "تشخيص", "طبيب", "سعال", "حمى",
    "ألم", "شفاء", "جرح", "حساسية", "حمل", "سكري", "غلوكوز", "أنسولين", "ميتفورمين",
    "كحول", "مناعة", "فيروس", "حليب", "عظام", "نبيذ", "مريض", "صحة", "تغذية",
    "ماء", "ترطيب", "سائل", "ثمانية أكواب", "نباتات", "فاكهة", "تمارين", "جفاف",
    "سرطان", "إشعاع", "قهوة", "فطور", "أرق", "قلق", "مناعة"
]

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")


def _get_groq_client():
    if load_dotenv is not None:
        load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or Groq is None:
        return None
    return Groq(api_key=api_key)


def _extract_claims_with_groq(text: str):
    client = _get_groq_client()
    if client is None:
        return None

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
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", GROQ_MODEL),
            temperature=0.1,
            reasoning_format="hidden",
            max_completion_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _strip_non_important_symbols(text)},
            ],
        )
        content = response.choices[0].message.content or "{}"
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
        if not payloads:
            return None
        payload = max(payloads, key=lambda item: len(item.get("claims", [])))
    except Exception:
        return None

    raw_claims = payload.get("claims", []) if isinstance(payload, dict) else []
    if not isinstance(raw_claims, list):
        return None

    claims = []
    seen = set()
    for item in raw_claims:
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
        claims.append(
            {
                "original_claim": original.rstrip(". "),
                "normalized_claim": normalized.rstrip(". "),
                "medical_query": query.rstrip(". "),
            }
        )
    return claims


def _contains_medical_context(text: str) -> bool:
    if text is None:
        return False
    lower = str(text).lower()
    return any(keyword.lower() in lower for keyword in MEDICAL_KEYWORDS)


def _strip_non_important_symbols(text: str) -> str:
    if text is None:
        return ""

    cleaned = str(text)
    cleaned = cleaned.replace("**", " ").replace("__", " ")
    cleaned = cleaned.replace("*", " ").replace("_", " ")
    cleaned = cleaned.replace("`", " ")
    cleaned = cleaned.replace("•", " ")
    cleaned = cleaned.replace("–", " ").replace("—", " ")
    cleaned = cleaned.replace("[", " ").replace("]", " ")
    cleaned = cleaned.replace("(", " ").replace(")", " ")
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = cleaned.replace("<", " ").replace(">", " ")
    cleaned = cleaned.replace("#", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .,:;!?/\\|\t\n")
    cleaned = re.sub(r"\s+-\s+", " ", cleaned)
    cleaned = re.sub(r"(?<=\w)- (?=\w)", " ", cleaned)
    return cleaned


def _split_sentences(text: str):
    text = _strip_non_important_symbols(text)
    text = re.sub(r"\s+", " ", text.strip())
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|(?<=[،])\s*", text) if part.strip()]


def _split_atomic_clauses(sentence: str):
    text = _strip_non_important_symbols(sentence)
    if not text:
        return []

    chunks = [text]
    if re.search(r"[\u0600-\u06FF]", text):
        chunks = re.split(r"\s+(?:و|لكن|حيث|كما|ولا)\s+", text)
    else:
        chunks = re.split(
            r"(?<=[.!?])\s+|(?<=[,;])\s+|\s+(?:and|or|but|which|where|while|however|therefore|thus|since|because|although)\s+",
            text,
            flags=re.IGNORECASE,
        )

    fragments = []
    for chunk in chunks:
        chunk = re.sub(r"^[\s,;:]+|[\s,;:]+$", "", str(chunk))
        if not chunk:
            continue

        cleaned = re.sub(r"\s+", " ", chunk).strip()
        if len(cleaned.split()) < 3:
            continue

        lower = cleaned.lower()
        if any(pattern in lower for pattern in [
            "let your body guide you",
            "i just wanted to clarify",
            "it can't be the same",
            "well, big, small",
            "whatnot",
            "so the point is",
            "you guessed it",
            "this tagline was invented",
        ]):
            continue

        if _contains_medical_context(cleaned):
            fragments.append(cleaned.rstrip(". "))
    return fragments


def generate_medical_query(claim: str) -> str:
    """Create a biomedical search query from a raw claim.

    The query stays in English for the downstream RAG layer, while the claim itself
    can remain Arabic or English.
    """
    if claim is None:
        return "clinical evidence for medical claim"

    cleaned = re.sub(r"\s+", " ", str(claim).strip())
    cleaned = cleaned.rstrip(". ")

    if not cleaned:
        return "clinical evidence for medical claim"

    if re.search(r"[\u0600-\u06FF]", cleaned):
        return f"Clinical evidence and safety of {cleaned}"

    query = cleaned
    if not any(keyword.lower() in query.lower() for keyword in ["evidence", "clinical", "treatment", "symptom", "infection", "diagnosis", "metformin", "diabetes", "insulin", "hypoglycemia"]):
        query = f"clinical evidence for {query}"
    return query


def extract_claims(text: str):
    """Extract claims with Groq, falling back to the local extractor when unavailable."""
    if text is None:
        return []

    cleaned_text = _strip_non_important_symbols(text)
    if not cleaned_text:
        return []

    llm_claims = _extract_claims_with_groq(cleaned_text)
    if llm_claims is not None:
        return llm_claims

    sentences = _split_sentences(cleaned_text)
    claims = []
    seen = set()

    for sentence in sentences:
        if not _contains_medical_context(sentence):
            continue

        candidate_clauses = _split_atomic_clauses(sentence)
        if not candidate_clauses:
            candidate_clauses = [sentence]

        for clause in candidate_clauses:
            clause_text = re.sub(r"\s+", " ", str(clause).strip())
            if not clause_text:
                continue
            if len(clause_text.split()) < 3:
                continue

            lower_clause = clause_text.lower()
            if any(pattern in lower_clause for pattern in [
                "let your body guide you",
                "i just wanted to clarify",
                "it can't be the same",
                "well, big, small",
                "whatnot",
                "so the point is",
                "you guessed it",
                "this tagline was invented",
            ]):
                continue
            if lower_clause in seen:
                continue
            seen.add(lower_clause)
            claims.append(
                {
                    "original_claim": clause_text,
                    "normalized_claim": clause_text,
                    "medical_query": generate_medical_query(clause_text),
                }
            )

    if not claims and cleaned_text and _contains_medical_context(cleaned_text):
        fallback = re.sub(r"\s+", " ", cleaned_text)
        if len(fallback.split()) >= 4:
            claims.append(
                {
                    "original_claim": fallback,
                    "normalized_claim": fallback,
                    "medical_query": generate_medical_query(fallback),
                }
            )

    return claims
