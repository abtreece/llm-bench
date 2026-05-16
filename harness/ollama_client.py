"""Thin HTTP wrapper around Ollama's /api/chat endpoint."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import requests


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 300
DEFAULT_NUM_CTX = 16384


@dataclass(frozen=True)
class ChatResult:
    content: str
    prompt_eval_count: int
    eval_count: int
    total_duration_ns: int


def chat(
    model: str,
    system: str,
    user: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> ChatResult:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
        },
    }
    r = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    return ChatResult(
        content=msg.get("content", ""),
        prompt_eval_count=int(data.get("prompt_eval_count", 0)),
        eval_count=int(data.get("eval_count", 0)),
        total_duration_ns=int(data.get("total_duration", 0)),
    )


def list_local_models(base_url: str = DEFAULT_BASE_URL) -> Iterable[str]:
    r = requests.get(f"{base_url}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]
