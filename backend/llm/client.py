"""Shared LLM client with Ollama-over-HTTP support and optional OpenAI fallback."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")


def _ollama_model(purpose: str = "default") -> str:
    if purpose == "query":
        return os.getenv("OLLAMA_QUERY_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b")).strip()
    if purpose == "verifier":
        return os.getenv("OLLAMA_VERIFIER_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:3b")).strip()
    return os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()


def _ollama_messages(system: str, user: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _debug(message: str) -> None:
    if os.getenv("MYTHLENS_LLM_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
        print(f"[MythLens LLM] {message}")


def _ollama_chat(system: str, user: str, json_mode: bool = False, purpose: str = "default") -> Optional[str]:
    base = _ollama_base_url()
    if not base:
        _debug("OLLAMA_BASE_URL is empty")
        return None

    payload: Dict[str, Any] = {
        "model": _ollama_model(purpose),
        "messages": _ollama_messages(system, user),
        "stream": False,
        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
        "options": {"temperature": 0},
    }
    if json_mode:
        payload["format"] = "json"

    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }

    timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
    try:
        response = requests.post(
            f"{base}/api/chat",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = str(data.get("message", {}).get("content", "")).strip()
        if not content:
            _debug(f"Ollama returned no message content: {data}")
        return content or None
    except Exception as exc:
        _debug(f"Ollama request failed ({purpose}/{_ollama_model(purpose)}): {type(exc).__name__}: {exc}")
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
    except Exception as exc:
        _debug(f"OpenAI fallback failed: {type(exc).__name__}: {exc}")
        return None


def chat_text(system: str, user: str, purpose: str = "default") -> Optional[str]:
    return _ollama_chat(system, user, json_mode=False, purpose=purpose) or _openai_chat(system, user, json_mode=False)


def chat_json(system: str, user: str, purpose: str = "default") -> Optional[Dict[str, Any]]:
    raw = _ollama_chat(system, user, json_mode=True, purpose=purpose) or _openai_chat(system, user, json_mode=True)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as exc:
        _debug(f"Could not parse LLM JSON: {type(exc).__name__}: {exc}; raw={raw[:500]!r}")
        return None
