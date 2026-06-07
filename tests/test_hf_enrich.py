"""Tests for the HuggingFace enrichment helper (ticket #12).

Uses unittest.mock.patch throughout — the real huggingface_hub package is
never imported (and may not be installed).  Tests verify the function's
behaviour under: happy path, unknown model, network error, missing dep,
caching, and the off-surface contract.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Off-surface: must NOT appear in __all__ and not importable from top level
# ---------------------------------------------------------------------------


def test_enrich_checkpoint_not_in_public_all():
    """enrich_checkpoint must not appear in lib_python_comfy.__all__."""
    import lib_python_comfy
    assert "enrich_checkpoint" not in lib_python_comfy.__all__


def test_enrich_checkpoint_not_importable_from_top_level():
    """from lib_python_comfy import enrich_checkpoint must raise ImportError."""
    with pytest.raises(ImportError):
        from lib_python_comfy import enrich_checkpoint  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(repo_id: str, tags: list[str] | None = None) -> MagicMock:
    """Return a MagicMock that looks like a huggingface_hub ModelInfo object."""
    m = MagicMock()
    m.id = repo_id
    m.modelId = repo_id
    m.tags = tags or []
    m.cardData = None
    return m


def _hf_hub_module(api_instance: MagicMock) -> MagicMock:
    """Return a fake huggingface_hub module whose HfApi() returns api_instance."""
    # Do not use spec=ModuleType — that restricts attributes to real ModuleType
    # attributes and would prevent setting HfApi.
    mod = MagicMock()
    mod.HfApi.return_value = api_instance
    return mod


# ---------------------------------------------------------------------------
# Clear lru_cache before each test so tests are independent
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the enrich_checkpoint lru_cache before every test."""
    # Import the cached implementation directly so we can clear it.
    from lib_python_comfy.hf_enrich import _cached_enrich
    _cached_enrich.cache_clear()
    yield
    _cached_enrich.cache_clear()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_enrich_checkpoint_happy_path():
    """With a valid model result the dict contains repo_id, tags, url."""
    api = MagicMock()
    api.list_models.return_value = [_make_mock_model("runwayml/stable-diffusion-v1-5", ["text-to-image"])]

    fake_hf = _hf_hub_module(api)

    with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
        from lib_python_comfy.hf_enrich import _cached_enrich
        _cached_enrich.cache_clear()

        from lib_python_comfy.hf_enrich import enrich_checkpoint
        result = enrich_checkpoint("stable-diffusion-v1-5")

    assert result["repo_id"] == "runwayml/stable-diffusion-v1-5"
    assert "text-to-image" in result["tags"]
    assert result["url"] == "https://huggingface.co/runwayml/stable-diffusion-v1-5"
    # All four keys must be present
    assert set(result.keys()) == {"repo_id", "tags", "summary", "url"}


# ---------------------------------------------------------------------------
# Unknown model — empty result list
# ---------------------------------------------------------------------------


def test_enrich_checkpoint_unknown_model_no_raise():
    """When list_models returns an empty list, repo_id is None and no exception is raised."""
    api = MagicMock()
    api.list_models.return_value = []

    fake_hf = _hf_hub_module(api)

    with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
        from lib_python_comfy.hf_enrich import _cached_enrich
        _cached_enrich.cache_clear()

        from lib_python_comfy.hf_enrich import enrich_checkpoint
        result = enrich_checkpoint("totally_unknown_model_xyz")

    assert result["repo_id"] is None
    assert result["tags"] == []
    assert result["url"] is None
    # Must not raise
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Network error
# ---------------------------------------------------------------------------


def test_enrich_checkpoint_network_error_no_raise():
    """A ConnectionError from list_models returns a safe dict without raising."""
    api = MagicMock()
    api.list_models.side_effect = ConnectionError("timeout")

    fake_hf = _hf_hub_module(api)

    with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
        from lib_python_comfy.hf_enrich import _cached_enrich
        _cached_enrich.cache_clear()

        from lib_python_comfy.hf_enrich import enrich_checkpoint
        result = enrich_checkpoint("any_model")

    assert isinstance(result, dict)
    # All four keys present
    assert "repo_id" in result
    assert "tags" in result
    assert "summary" in result
    assert "url" in result


# ---------------------------------------------------------------------------
# Missing dependency
# ---------------------------------------------------------------------------


def test_enrich_checkpoint_missing_dep_returns_empty_dict():
    """When huggingface_hub is not installed, enrich_checkpoint returns {}."""
    # Simulate missing module by setting sys.modules entry to None
    with patch.dict(sys.modules, {"huggingface_hub": None}):
        from lib_python_comfy.hf_enrich import _cached_enrich
        _cached_enrich.cache_clear()

        from lib_python_comfy.hf_enrich import enrich_checkpoint
        result = enrich_checkpoint("any_model")

    assert result == {}


# ---------------------------------------------------------------------------
# Caching — underlying API called only once for identical args
# ---------------------------------------------------------------------------


def test_enrich_checkpoint_caching():
    """Repeated calls with the same name hit lru_cache; HfApi.list_models called once."""
    api = MagicMock()
    api.list_models.return_value = [_make_mock_model("org/model")]

    fake_hf = _hf_hub_module(api)

    with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
        from lib_python_comfy.hf_enrich import _cached_enrich
        _cached_enrich.cache_clear()

        from lib_python_comfy.hf_enrich import enrich_checkpoint
        r1 = enrich_checkpoint("org/model")
        r2 = enrich_checkpoint("org/model")
        r3 = enrich_checkpoint("org/model")

    # list_models should have been called exactly once despite three calls
    assert api.list_models.call_count == 1
    assert r1 == r2 == r3
