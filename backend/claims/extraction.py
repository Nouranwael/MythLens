import re

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
    """Extract atomic medical claims from a transcript or text input."""
    if text is None:
        return []

    sentences = _split_sentences(text)
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

    if not claims and text.strip() and _contains_medical_context(text):
        fallback = re.sub(r"\s+", " ", str(text).strip())
        if len(fallback.split()) >= 4:
            claims.append(
                {
                    "original_claim": fallback,
                    "normalized_claim": fallback,
                    "medical_query": generate_medical_query(fallback),
                }
            )

    return claims
