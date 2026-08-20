"""Shared Gemini REST client for MythLens.

Uses the Interactions API directly with the x-goog-api-key header so both
standard and authorization (AQ...) Gemini API keys work without SDK auth
translation issues.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests

GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def _debug(message: str) -> None:
    if os.getenv("MYTHLENS_LLM_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
        print(f"[MythLens LLM] {message}")


def _model_for(purpose: str) -> str:
    default_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
    purpose = (purpose or "general").strip().lower()
    if purpose == "query":
        return os.getenv("GEMINI_QUERY_MODEL", default_model).strip() or default_model
    if purpose in {"verify", "verifier", "verification"}:
        return os.getenv("GEMINI_VERIFIER_MODEL", default_model).strip() or default_model
    return default_model


def _extract_interaction_text(data: Dict[str, Any]) -> Optional[str]:
    texts = []
    for step in data.get("steps", []) or []:
        if step.get("type") != "model_output":
            continue
        for item in step.get("content", []) or []:
            if item.get("type") == "text" and item.get("text"):
                texts.append(str(item["text"]))
    text = "\n".join(texts).strip()
    return text or None


def _gemini_chat(
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    purpose: str = "general",
) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        _debug("GEMINI_API_KEY is empty")
        return None

    payload: Dict[str, Any] = {
        "model": _model_for(purpose),
        "input": user,
        "system_instruction": system,
        "store": False,
        "generation_config": {
            "thinking_level": "minimal",
            "max_output_tokens": 256 if purpose == "query" else 1024,
        },
    }
    if json_mode:
        payload["response_format"] = {
            "type": "text",
            "mime_type": "application/json",
        }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    try:
        response = requests.post(
            GEMINI_INTERACTIONS_URL,
            headers=headers,
            json=payload,
            timeout=90,
        )
        if not response.ok:
            _debug(f"Gemini REST failed: HTTP {response.status_code}: {response.text[:800]}")
            return None
        data = response.json()
        text = _extract_interaction_text(data)
        if not text:
            _debug(f"Gemini REST returned no model text: {data}")
        return text
    except Exception as exc:
        _debug(f"Gemini REST request failed: {type(exc).__name__}: {exc}")
        return None


def chat_text(system: str, user: str, *, purpose: str = "general") -> Optional[str]:
    return _gemini_chat(system, user, json_mode=False, purpose=purpose)


def chat_json(system: str, user: str, *, purpose: str = "verifier") -> Optional[Dict[str, Any]]:
    raw = _gemini_chat(system, user, json_mode=True, purpose=purpose)
    if not raw:
        return None

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "", 1).strip()

    try:
        return json.loads(cleaned)
    except Exception as exc:
        _debug(f"Could not parse Gemini JSON: {type(exc).__name__}: {exc}; raw={raw[:500]!r}")
        return None
