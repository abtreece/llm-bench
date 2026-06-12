"""Tests for harness/recommend.py — pure-function core, no subprocess/network."""
from harness import ollama_client


class TestModelsFromTags:
    def test_converts_bytes_to_decimal_gb(self):
        data = {"models": [{"name": "qwen2.5:14b", "size": 9_000_000_000}]}
        assert ollama_client.models_from_tags(data) == [
            {"name": "qwen2.5:14b", "size_gb": 9.0}
        ]

    def test_empty_and_missing_models_key(self):
        assert ollama_client.models_from_tags({}) == []
        assert ollama_client.models_from_tags({"models": []}) == []
