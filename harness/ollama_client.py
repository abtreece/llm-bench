"""Thin HTTP wrapper around Ollama's /api/chat endpoint."""
from __future__ import annotations

from dataclasses import dataclass

import requests


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 300
DEFAULT_NUM_CTX = 16384


class ChatTimeout(Exception):
    """The model did not finish generating within the read timeout.

    Distinct from connection-level failures: on fixed hardware this is a
    property of the model (too slow), so the harness scores it as a model
    outcome rather than excluding the row as infra noise.
    """


@dataclass(frozen=True)
class ChatResult:
    content: str
    # Reasoning emitted by thinking models (deepseek-r1, qwen3). Ollama >= 0.9
    # routes it to message.thinking, so content stays parseable; empty for
    # non-thinking models. Thinking tokens are included in eval_count.
    thinking: str
    prompt_eval_count: int
    eval_count: int
    total_duration_ns: int
    load_duration_ns: int
    eval_duration_ns: int
    done_reason: str  # "stop" = natural end; "length" = truncated by token limit


def chat(
    model: str,
    system: str,
    user: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    num_ctx: int = DEFAULT_NUM_CTX,
    temperature: float = 0.0,
    seed: int | None = None,
) -> ChatResult:
    options = {
        "temperature": temperature,
        "num_ctx": num_ctx,
    }
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": options,
    }
    # (connect, read) — read covers the full non-streamed generation.
    try:
        r = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=(10, timeout_s),
        )
    except requests.exceptions.ReadTimeout as e:
        raise ChatTimeout(f"no response from {model} after {timeout_s}s") from e
    r.raise_for_status()
    return chat_result_from_response(r.json())


def chat_result_from_response(data: dict) -> ChatResult:
    """Pure extraction from an /api/chat payload into a ChatResult."""
    msg = data.get("message") or {}
    return ChatResult(
        content=msg.get("content", ""),
        # `or ""`: absent for non-thinking models; null-tolerant either way.
        thinking=msg.get("thinking") or "",
        prompt_eval_count=int(data.get("prompt_eval_count", 0)),
        eval_count=int(data.get("eval_count", 0)),
        total_duration_ns=int(data.get("total_duration", 0)),
        load_duration_ns=int(data.get("load_duration", 0)),
        eval_duration_ns=int(data.get("eval_duration", 0)),
        done_reason=str(data.get("done_reason", "")),
    )


def warm_up(
    model: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> None:
    """Load the model into memory so benchmark latencies exclude cold-start."""
    chat(model, "You are a helpful assistant.", "Reply with: ok",
         base_url=base_url, timeout_s=timeout_s, num_ctx=512)


def models_from_tags(data: dict) -> list[dict]:
    """Pure extraction from an /api/tags payload:
    [{"name", "size_gb", "digest", "families"}].

    Ollama reports size in bytes; GB = bytes / 1e9 to match the registry's
    decimal-GB convention used in models.yaml and catalog.yaml. digest is
    the manifest sha256 (bare hex), shared by all tags of the same model.
    """
    models: list[dict] = []
    # `or []`: Ollama's Go server marshals a nil slice as {"models": null}.
    for m in data.get("models") or []:
        details = m.get("details") or {}
        # Older payloads carry "family" with "families": null.
        families = details.get("families") or (
            [details["family"]] if details.get("family") else [])
        models.append({
            "name": m["name"],
            "size_gb": int(m.get("size", 0)) / 1e9,
            "digest": str(m.get("digest") or ""),
            "families": list(families),
        })
    return models


def list_local_models(base_url: str = DEFAULT_BASE_URL) -> list[dict]:
    r = requests.get(f"{base_url}/api/tags", timeout=10)
    r.raise_for_status()
    return models_from_tags(r.json())
