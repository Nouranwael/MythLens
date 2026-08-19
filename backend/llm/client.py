"""Shared Gemini LLM client for MythLens."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def _debug(message: str) -> None:
    if os.getenv("MYTHLENS_LLM_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
        print(f"[MythLens LLM] {message}")


def _model_for(purpose: str) -> str:
    default_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    purpose = (purpose or "general").strip().lower()
    if purpose == "query":
        return os.getenv("GEMINI_QUERY_MODEL", default_model).strip() or default_model
    if purpose in {"verify", "verifier", "verification"}:
        return os.getenv("GEMINI_VERIFIER_MODEL", default_model).strip() or default_model
    return default_model


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

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        max_tokens = 256 if purpose == "query" else 1024
        config_kwargs: Dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        response = client.models.generate_content(
            model=_model_for(purpose),
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = (response.text or "").strip()
        if not text:
            _debug("Gemini returned an empty response")
            return None
        return text
    except Exception as exc:
        _debug(f"Gemini request failed: {type(exc).__name__}: {exc}")
        return None


def chat_text(system: str, user: str, *, purpose: str = "general") -> Optional[str]:
    return _gemini_chat(system, user, json_mode=False, purpose=purpose)


def chat_json(system: str, user: str, *, purpose: str = "verifier") -> Optional[Dict[str, Any]]:
    raw = _gemini_chat(system, user, json_mode=True, purpose=purpose)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as exc:
        _debug(f"Could not parse Gemini JSON: {type(exc).__name__}: {exc}; raw={raw[:500]!r}")
        return None
