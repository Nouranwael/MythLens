"""Shared Gemini LLM client for MythLens using the current Interactions API."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


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

        client = genai.Client(api_key=api_key)

        prompt = f"SYSTEM INSTRUCTIONS:\n{system.strip()}\n\nUSER INPUT:\n{user.strip()}"
        if json_mode:
            prompt += "\n\nReturn ONLY one valid JSON object. Do not use markdown fences or add any text before or after the JSON."

        interaction = client.interactions.create(
            model=_model_for(purpose),
            input=prompt,
        )
        text = (interaction.output_text or "").strip()
        if not text:
            _debug("Gemini Interactions API returned an empty response")
            return None
        return text
    except Exception as exc:
        _debug(f"Gemini Interactions request failed: {type(exc).__name__}: {exc}")
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
