"""HuggingFace model-card enrichment (off-surface helper).

This module is intentionally **not** re-exported from
``lib_python_comfy.__init__`` and is **not** part of ``__all__``.  It is a
best-effort utility that requires the optional ``huggingface_hub`` dependency.

Usage::

    from lib_python_comfy.hf_enrich import enrich_checkpoint
    info = enrich_checkpoint("runwayml/stable-diffusion-v1-5")

The function never raises; it returns an empty dict (or a partial dict) when
the dependency is absent or a network/API error occurs.
"""
from __future__ import annotations

import functools


def enrich_checkpoint(name: str, *, sha256: str | None = None) -> dict:
    """Return HuggingFace metadata for a checkpoint by name.

    Performs a best-effort ``list_models`` search via the HuggingFace Hub API.
    The result is cached with :func:`functools.lru_cache` (keyed on *name* and
    *sha256*) so repeated calls for the same checkpoint are free.

    Parameters
    ----------
    name:
        Checkpoint name or HuggingFace repo id to search for.
    sha256:
        Optional SHA-256 hash (currently unused in the search; reserved for
        future exact-match lookups).

    Returns:
        A dict with the following keys (all always present):

        .. code-block:: python

            {
                "repo_id": str | None,
                "tags":    list[str],
                "summary": str | None,
                "url":     str | None,
            }

        Returns ``{}`` immediately when ``huggingface_hub`` is not installed.
        Returns a dict with ``None``/empty values on network or API errors.
    """
    return _cached_enrich(name, sha256)


@functools.lru_cache(maxsize=256)
def _cached_enrich(name: str, sha256: str | None) -> dict:
    """Cached implementation of :func:`enrich_checkpoint`."""
    try:
        import huggingface_hub  # noqa: PLC0415
    except ImportError:
        return {}

    _empty: dict = {"repo_id": None, "tags": [], "summary": None, "url": None}

    try:
        api = huggingface_hub.HfApi()
        models = list(api.list_models(search=name, limit=5))
        if not models:
            return _empty

        # Pick the first result as the best match.
        best = models[0]
        repo_id: str | None = getattr(best, "id", None) or getattr(best, "modelId", None)
        tags: list[str] = list(getattr(best, "tags", None) or [])

        # Card summary — may be None if no model card exists.
        summary: str | None = None
        card_data = getattr(best, "cardData", None)
        if card_data and isinstance(card_data, dict):
            summary = card_data.get("description") or card_data.get("summary")

        url: str | None = None
        if repo_id:
            url = f"https://huggingface.co/{repo_id}"

        return {"repo_id": repo_id, "tags": tags, "summary": summary, "url": url}

    except Exception:  # noqa: BLE001 — network/rate-limit/unknown, always return dict
        return _empty
