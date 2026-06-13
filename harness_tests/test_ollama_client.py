"""Tests for harness/ollama_client.py — pure response extraction.

Run explicitly (pyproject's testpaths keeps these out of the benchmark
suite that models are graded against):

    .venv/bin/pytest harness_tests/
"""
from harness import ollama_client


def chat_payload(**overrides) -> dict:
    payload = {
        "message": {"role": "assistant", "content": "ok"},
        "prompt_eval_count": 10,
        "eval_count": 20,
        "total_duration": 3_000_000_000,
        "load_duration": 100_000_000,
        "eval_duration": 2_000_000_000,
        "done_reason": "stop",
    }
    payload.update(overrides)
    return payload


class TestChatResultFromResponse:
    def test_non_thinking_model_has_empty_thinking(self):
        result = ollama_client.chat_result_from_response(chat_payload())
        assert result.content == "ok"
        assert result.thinking == ""
        assert result.prompt_eval_count == 10
        assert result.eval_count == 20
        assert result.done_reason == "stop"

    def test_thinking_model_separates_reasoning_from_content(self):
        payload = chat_payload(
            message={
                "role": "assistant",
                "content": "```python\n# path: app/money.py\nx = 1\n```",
                "thinking": "Let me reason about the bug first.",
            }
        )
        result = ollama_client.chat_result_from_response(payload)
        assert result.thinking == "Let me reason about the bug first."
        assert "# path: app/money.py" in result.content
        assert "reason about" not in result.content

    def test_null_thinking_field_tolerated(self):
        payload = chat_payload(message={"role": "assistant", "content": "ok", "thinking": None})
        result = ollama_client.chat_result_from_response(payload)
        assert result.thinking == ""

    def test_missing_fields_default_to_zero(self):
        result = ollama_client.chat_result_from_response({"message": None})
        assert result.content == ""
        assert result.thinking == ""
        assert result.eval_count == 0
        assert result.done_reason == ""
