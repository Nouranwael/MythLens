import re

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: str) -> str:
    """Detect whether the input is Egyptian Arabic or English."""
    if not text or not str(text).strip():
        return "en-US"
    if ARABIC_RANGE.search(text):
        return "ar-EG"
    return "en-US"


def normalize_egyptian_arabic(text: str) -> str:
    """Normalize Egyptian Arabic text and clean repeated spacing."""
    if text is None:
        return ""

    text = str(text).strip()
    if not text:
        return ""

    replacements = {
        "\u0640": "",
        "\u200F": "",
        "\u200E": "",
        "\u00A0": " ",
        "  ": " ",
        "،": ",",
        "؛": ";",
        "؟": "?",
        "!": "! ",
        "…": "...",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,;.!?])", r"\1", text)
    return text.strip()


def summarize_key_points(text: str, language: str = "en-US", max_sentences: int = 3) -> str:
    """Return the most important sentences from a transcript.

    This intentionally stays lightweight and dependency-free so AIHachthon teams can
    prototype without installing a large NLP stack.
    """
    if text is None:
        return ""

    cleaned = str(text).strip()
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+|(?<=[،])\s*", cleaned)
    sentences = [s.strip() for s in sentences if s and s.strip()]

    if not sentences:
        return cleaned

    medical_keywords = {
        "en-US": ["treatment", "symptom", "infection", "doctor", "medicine", "diagnosis", "cure", "healing", "cough", "fever", "pain", "risk", "pregnant", "allergy"],
        "ar-EG": ["علاج", "أعراض", "عدوى", "طبيب", "دواء", "تشخيص", "شفاء", "سعال", "حمى", "ألم", "خطر", "حمل", "حساسية"],
    }

    keyword_list = medical_keywords.get(language, medical_keywords["en-US"])
    scored = []
    for sentence in sentences:
        lower = sentence.lower()
        keyword_hits = sum(1 for keyword in keyword_list if keyword.lower() in lower)
        words = len(sentence.split())
        score = keyword_hits * 3 + max(0, words - 8)
        scored.append((score, sentence))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [sentence for _, sentence in scored[:max_sentences]]
        return " ".join(top)

    return " ".join(sentences[:max_sentences])
