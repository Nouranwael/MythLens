"""Medical claim extraction and retrieval-query generation."""

from __future__ import annotations

import re

from backend.llm.client import chat_text

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

ARABIC_MEDICAL_GLOSSARY = {
    "الثوم": "garlic", "ثوم": "garlic", "الليمون": "lemon", "ليمون": "lemon",
    "العسل": "honey", "عسل": "honey", "الصيام": "fasting", "صيام": "fasting",
    "المعدة": "stomach", "معدة": "stomach", "القولون": "colon irritable bowel",
    "قولون": "colon irritable bowel", "السكري": "diabetes mellitus", "سكري": "diabetes mellitus",
    "السكر": "blood glucose diabetes", "أنسولين": "insulin", "الأنسولين": "insulin",
    "الضغط": "blood pressure hypertension", "ضغط": "blood pressure hypertension",
    "السرطان": "cancer", "سرطان": "cancer", "المناعة": "immune system", "مناعة": "immune system",
    "جرح": "wound", "الجروح": "wounds", "حرق": "burn", "حروق": "burns",
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


def _llm_biomedical_query(arabic_claim: str):
    query = chat_text(
        "Convert an Egyptian/Arabic medical claim into a concise English biomedical PubMed search query. Preserve intervention, condition, claimed effect, population, dose and duration when present. Output only the query.",
        arabic_claim,
    )
    if query:
        query = query.strip().strip('"')
        if query and not ARABIC_RANGE.search(query):
            return re.sub(r"\s+", " ", query)
    return None


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
    if ARABIC_RANGE.search(cleaned):
        return _llm_biomedical_query(cleaned) or _fallback_arabic_query(cleaned)
    if not any(keyword in cleaned.lower() for keyword in ["evidence", "clinical", "treatment", "symptom", "infection", "diagnosis", "metformin", "diabetes", "insulin", "hypoglycemia"]):
        return f"clinical evidence for {cleaned}"
    return cleaned


def extract_claims(text: str):
    if text is None:
        return []
    claims, seen = [], set()
    for sentence in _split_sentences(text):
        if not _contains_medical_context(sentence):
            continue
        for clause in (_split_atomic_clauses(sentence) or [sentence]):
            clause_text = re.sub(r"\s+", " ", str(clause).strip())
            lower = clause_text.lower()
            if len(clause_text.split()) < 3 or lower in seen:
                continue
            seen.add(lower)
            claims.append({"original_claim": clause_text, "normalized_claim": clause_text, "medical_query": generate_medical_query(clause_text)})
    if not claims and str(text).strip() and _contains_medical_context(text):
        fallback = re.sub(r"\s+", " ", str(text).strip())
        if len(fallback.split()) >= 4:
            claims.append({"original_claim": fallback, "normalized_claim": fallback, "medical_query": generate_medical_query(fallback)})
    return claims
