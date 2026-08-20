"""Live PubMed retrieval through NCBI E-utilities."""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

import requests

from .config import NCBI_API_KEY, NCBI_EMAIL, PUBMED_BASE

SESSION = requests.Session()
REQUEST_DELAY = 0.12 if NCBI_API_KEY else 0.40
MAX_RETRIES = 3

# Conservative biomedical synonym expansions used only to improve retrieval recall.
# They do not change the medical claim or the final verdict.
PHRASE_EXPANSIONS = {
    "gastric ulcer": ["peptic ulcer", "stomach ulcer"],
    "irritable bowel syndrome": ["IBS"],
    "common cold": ["upper respiratory infection", "upper respiratory tract infection"],
    "type 2 diabetes": ["type 2 diabetes mellitus", "T2DM"],
    "wound healing": ["wounds", "wound treatment"],
    "dehydration": ["hydration", "fluid balance"],
    "microwave oven radiation": ["microwave radiation", "radiofrequency electromagnetic fields"],
    "detox drinks": ["detoxification", "detox diet"],
    "antibiotic replacement": ["antibiotic alternative"],
}

QUERY_FILLER = {
    "evidence", "safety", "safe", "required", "requirement", "claim", "claims",
    "treatment", "treat", "treats", "causes", "cause", "prevents", "prevent",
    "replacement", "replace", "removes", "remove", "effect", "effects", "efficacy",
    "habitual", "consumption", "daily", "per", "day", "body", "can", "does", "do",
    "is", "are", "help", "could", "would", "may", "new", "study", "studies",
    "review", "reviews", "report", "reports", "reported", "find", "finds", "found",
    "says", "say", "shows", "show", "experts", "issue", "issued", "gets", "get",
    "promising", "breakthrough", "major", "big", "deal", "mesh", "sh", "and", "or", "not",
}


def _params(**kwargs):
    params = dict(kwargs)
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    params["tool"] = "MythLens"
    return params


def safe_get(url, params):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            response = SESSION.get(url, params=params, timeout=30)
            if response.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
    raise last_error or RuntimeError("PubMed request failed")


def build_pubmed_query(medical_query):
    query = re.sub(r"[?!.]", " ", str(medical_query).lower()).strip()
    stop = {"can", "does", "do", "is", "are", "help", "could", "would", "may"}
    query = " ".join(w for w in query.split() if w not in stop)
    replacements = {
        "high blood pressure": "high blood pressure hypertension",
        "heart attack": "heart attack myocardial infarction",
        "high cholesterol": "high cholesterol hypercholesterolemia",
    }
    for old, new in replacements.items():
        if old in query:
            query = query.replace(old, new)
    if "diabetes" in query and "diabetes mellitus" not in query:
        query = query.replace("diabetes", "diabetes mellitus")
    words = []
    for word in query.split():
        if not words or word != words[-1]:
            words.append(word)
    return " ".join(words)


def simplify_medical_query(query):
    query = re.sub(r"[^\w\s-]", " ", str(query).lower())
    return " ".join(w for w in query.split() if w not in QUERY_FILLER)


def _clean_pubmed_term(value):
    value = re.sub(r"\[[^\]]+\]", " ", str(value))
    value = re.sub(r"[\"'()]", " ", value)
    value = re.sub(r"\b(?:AND|OR|NOT)\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" -")
    return value


def _phrase_expansion_queries(raw_query):
    """Return broad synonym variants while preserving the main medical concepts."""
    lower = _clean_pubmed_term(raw_query).lower()
    variants = []
    for phrase, synonyms in PHRASE_EXPANSIONS.items():
        if phrase not in lower:
            continue
        for synonym in synonyms:
            candidate = lower.replace(phrase, synonym)
            candidate = simplify_medical_query(candidate)
            if candidate and candidate not in variants:
                variants.append(candidate)
    return variants


def _core_concept_queries(raw_query):
    """Build broad noun-focused fallbacks for natural-language fact-check queries."""
    cleaned = simplify_medical_query(_clean_pubmed_term(raw_query))
    tokens = [token for token in cleaned.split() if len(token) > 2]
    variants = []
    if cleaned:
        variants.append(cleaned)
    # Smaller concept windows improve recall when the full assertion is too strict.
    for size in (4, 3, 2):
        if len(tokens) >= size:
            candidate = " ".join(tokens[:size])
            if candidate and candidate not in variants:
                variants.append(candidate)
    return variants


def build_relaxed_queries(query):
    """Create progressively broader fallbacks for MeSH/Boolean or natural queries."""
    raw = str(query or "").strip()
    if not raw:
        return []

    relaxed = []
    groups = re.split(r"\s+AND\s+", raw, flags=re.IGNORECASE)
    concepts = []
    for group in groups:
        alternatives = re.split(r"\s+OR\s+", group, flags=re.IGNORECASE)
        first = _clean_pubmed_term(alternatives[0] if alternatives else group)
        first = simplify_medical_query(first)
        if first and first not in concepts:
            concepts.append(first)

    if len(concepts) > 1:
        for size in (2, 3, len(concepts)):
            if len(concepts) >= size:
                candidate = " ".join(concepts[:size]).strip()
                if candidate and candidate not in relaxed:
                    relaxed.append(candidate)

    for candidate in _core_concept_queries(raw):
        if candidate not in relaxed:
            relaxed.append(candidate)
    for candidate in _phrase_expansion_queries(raw):
        if candidate not in relaxed:
            relaxed.append(candidate)
    return relaxed


def search_pubmed(query, retmax=20):
    response = safe_get(
        f"{PUBMED_BASE}/esearch.fcgi",
        _params(db="pubmed", term=query, retmode="json", retmax=retmax, sort="relevance"),
    )
    return response.json().get("esearchresult", {}).get("idlist", [])


def search_pubmed_multi_query(medical_query, retmax_per_query=20, max_queries=6):
    """Search several query variants and fuse unique PMIDs.

    At least multiple variants are attempted instead of stopping as soon as the
    first query fills ``retmax``. This improves recall for myth-like natural
    language assertions that PubMed may parse too literally.
    """
    queries = [
        medical_query,
        build_pubmed_query(medical_query),
        *build_relaxed_queries(medical_query),
    ]
    unique_queries = []
    for query in queries:
        query = str(query).strip()
        if query and query.casefold() not in {q.casefold() for q in unique_queries}:
            unique_queries.append(query)
        if len(unique_queries) >= max_queries:
            break

    all_pmids = []
    for query in unique_queries:
        try:
            for pmid in search_pubmed(query, retmax=retmax_per_query):
                if pmid not in all_pmids:
                    all_pmids.append(pmid)
        except Exception:
            continue
    return all_pmids


def _text(element):
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def fetch_pubmed_articles(pmids):
    if not pmids:
        return []
    response = safe_get(
        f"{PUBMED_BASE}/efetch.fcgi",
        _params(db="pubmed", id=",".join(pmids), retmode="xml"),
    )
    root = ET.fromstring(response.content)
    articles = []
    for record in root.findall(".//PubmedArticle"):
        pmid = _text(record.find(".//PMID"))
        title = _text(record.find(".//ArticleTitle"))
        abstract_parts = [_text(el) for el in record.findall(".//Abstract/AbstractText")]
        abstract = " ".join(part for part in abstract_parts if part).strip()
        publication_types = [_text(el) for el in record.findall(".//PublicationType") if _text(el)]
        articles.append({
            "id": f"pubmed-{pmid}",
            "pmid": pmid,
            "title": title,
            "text": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            "publication_types": publication_types,
        })
    return articles


def remove_retracted_articles(articles):
    return [
        article for article in articles
        if "retracted publication" not in " ".join(article.get("publication_types", [])).lower()
    ]


def get_best_study_type(publication_types):
    text = " ".join(publication_types or []).lower()
    order = [
        ("meta-analysis", "Meta-Analysis"),
        ("systematic review", "Systematic Review"),
        ("randomized controlled trial", "Randomized Controlled Trial"),
        ("clinical trial", "Clinical Trial"),
        ("observational study", "Observational Study"),
        ("case reports", "Case Report"),
        ("review", "Review"),
        ("journal article", "Journal Article"),
    ]
    for token, label in order:
        if token in text:
            return label
    return publication_types[0] if publication_types else "Unknown"
