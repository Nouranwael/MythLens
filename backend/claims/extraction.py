import re

MEDICAL_KEYWORDS = [
    "treatment", "cure", "healing", "medicine", "medication", "symptom",
    "infection", "diagnosis", "doctor", "hospital", "viral", "bacterial",
    "pain", "fever", "cough", "remedy", "wound", "allergy", "pregnant",
    "metformin", "diabetes", "insulin", "gluconeogenesis", "hypoglycemia",
    "type 2", "glucose", "glycemic", "therapy", "clinical", "drug", "pharmacological",
    "علاج", "دواء", "أعراض", "عدوى", "تشخيص", "طبيب", "سعال", "حمى",
    "ألم", "شفاء", "جرح", "حساسية", "حمل", "سكري", "غلوكوز", "أنسولين", "ميتفورمين"
]


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
            r"(?<=[,;])\s+|\s+(?:and|or|but|which|where|while|however|therefore|thus|since|because|although)\s+",
            text,
            flags=re.IGNORECASE,
        )

    fragments = []
    for chunk in chunks:
        chunk = re.sub(r"^[\s,;:]+|[\s,;:]+$", "", str(chunk))
        if not chunk:
            continue
        if len(chunk.split()) < 3:
            continue
        if any(keyword.lower() in chunk.lower() for keyword in MEDICAL_KEYWORDS):
            fragments.append(chunk.rstrip(". "))
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
        lower = sentence.lower()
        if not any(keyword.lower() in lower for keyword in MEDICAL_KEYWORDS):
            continue

        candidate_clauses = _split_atomic_clauses(sentence)
        if not candidate_clauses:
            candidate_clauses = [sentence]

        for clause in candidate_clauses:
            clause_text = re.sub(r"\s+", " ", str(clause).strip())
            if not clause_text:
                continue
            if len(clause_text.split()) < 4:
                continue
            if clause_text.lower() in seen:
                continue
            seen.add(clause_text.lower())
            claims.append(
                {
                    "original_claim": clause_text,
                    "normalized_claim": clause_text,
                    "medical_query": generate_medical_query(clause_text),
                }
            )

    if not claims and text.strip():
        fallback = re.sub(r"\s+", " ", str(text).strip())
        claims.append(
            {
                "original_claim": fallback,
                "normalized_claim": fallback,
                "medical_query": generate_medical_query(fallback),
            }
        )

    return claims
