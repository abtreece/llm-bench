"""Tests for harness/recommend.py — pure-function core, no subprocess/network."""
from harness import ollama_client, recommend


class TestModelsFromTags:
    def test_converts_bytes_to_decimal_gb(self):
        data = {"models": [{"name": "qwen2.5:14b", "size": 9_000_000_000}]}
        assert ollama_client.models_from_tags(data) == [
            {"name": "qwen2.5:14b", "size_gb": 9.0}
        ]

    def test_empty_and_missing_models_key(self):
        assert ollama_client.models_from_tags({}) == []
        assert ollama_client.models_from_tags({"models": []}) == []


class TestCatalog:
    def test_repo_catalog_loads_and_is_well_formed(self):
        entries = recommend.load_catalog()
        assert len(entries) >= 8
        for e in entries:
            assert isinstance(e["name"], str) and ":" in e["name"]
            assert e["size_gb"] > 0

    def test_missing_catalog_degrades_to_empty(self, tmp_path):
        assert recommend.load_catalog(tmp_path / "nope.yaml") == []
