"""Shared LLM client with Ollama-over-HTTP support and optional OpenAI fallback."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")


def _ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()


def _ollama_messages(system: str, user: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _ollama_chat(system: str, user: str, json_mode: bool = False) -> Optional[str]:
    base = _ollama_base_url()
    if not base:
        return None

    payload: Dict[str, Any] = {
        "model": _ollama_model(),
        "messages": _ollama_messages(system, user),
        "stream": False,
        "options": {"temperature": 0},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        response = requests.post(f"{base}/api/chat", json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        content = str(data.get("message", {}).get("content", "")).strip()
        return content or None
    except Exception:
        return None


def _openai_chat(system: str, user: str, json_mode: bool = False) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip() or None
    except Exception:
        return None


def chat_text(system: str, user: str) -> Optional[str]:
    return _ollama_chat(system, user, json_mode=False) or _openai_chat(system, user, json_mode=False)


def chat_json(system: str, user: str) -> Optional[Dict[str, Any]]:
    raw = _ollama_chat(system, user, json_mode=True) or _openai_chat(system, user, json_mode=True)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None
