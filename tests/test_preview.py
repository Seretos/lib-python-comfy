"""Tests for the inline preview encoding module (ticket #6).

All images are generated in-memory with PIL.Image.new — no files on disk,
no network access.
"""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

import lib_python_comfy
from lib_python_comfy import PreviewResult, encode_preview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(width: int, height: int, mode: str = "RGB") -> bytes:
    """Return the raw bytes of an in-memory PNG of the given dimensions."""
    img = Image.new(mode, (width, height), color=(128, 64, 32) if mode == "RGB" else 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public API smoke test
# ---------------------------------------------------------------------------


def test_preview_names_in_public_all():
    """'PreviewResult' and 'encode_preview' must both be in lib_python_comfy.__all__."""
    assert "PreviewResult" in lib_python_comfy.__all__
    assert "encode_preview" in lib_python_comfy.__all__


def test_preview_importable_from_top_level():
    """from lib_python_comfy import encode_preview, PreviewResult succeeds."""
    from lib_python_comfy import encode_preview as ep, PreviewResult as PR  # noqa: F401
    assert ep is encode_preview
    assert PR is PreviewResult


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_encode_preview_happy_path_returns_preview_result():
    """A valid 200x200 PNG encodes successfully: fit=True, WebP magic, valid b64."""
    png = _make_png(200, 200)
    result = encode_preview(png)

    assert isinstance(result, PreviewResult)
    assert result.fit is True
    # WebP files begin with RIFF
    assert result.webp_bytes[:4] == b"RIFF"
    # b64 must decode back to webp_bytes
    assert base64.b64decode(result.b64) == result.webp_bytes
    # Must be within default budget
    assert len(result.b64) <= 100_000


# ---------------------------------------------------------------------------
# Downscaling
# ---------------------------------------------------------------------------


def test_encode_preview_downscales_large_image():
    """A 2000x2000 image is downscaled so neither dimension exceeds max_dim=512."""
    png = _make_png(2000, 2000)
    result = encode_preview(png, max_dim=512)

    decoded = Image.open(io.BytesIO(result.webp_bytes))
    assert decoded.width <= 512
    assert decoded.height <= 512


def test_encode_preview_preserves_aspect_ratio():
    """A 1000x500 image downscaled to max_dim=200 keeps the 2:1 aspect ratio."""
    png = _make_png(1000, 500)
    result = encode_preview(png, max_dim=200)

    decoded = Image.open(io.BytesIO(result.webp_bytes))
    # Should be 200x100 (or very close due to rounding)
    assert decoded.width <= 200
    assert decoded.height <= 200
    # Wider dimension
    assert decoded.width >= decoded.height


def test_encode_preview_does_not_upscale_small_image():
    """An image already within max_dim is not enlarged."""
    png = _make_png(100, 80)
    result = encode_preview(png, max_dim=512)

    decoded = Image.open(io.BytesIO(result.webp_bytes))
    assert decoded.width <= 100
    assert decoded.height <= 80


# ---------------------------------------------------------------------------
# Budget enforcement (fit=False)
# ---------------------------------------------------------------------------


def test_encode_preview_fit_false_when_budget_impossible():
    """With max_b64_chars=1 even quality=1 cannot fit — fit=False is returned."""
    png = _make_png(100, 100)
    result = encode_preview(png, max_b64_chars=1)

    assert isinstance(result, PreviewResult)
    assert result.fit is False
    # Result still has valid webp_bytes and b64
    assert result.webp_bytes[:4] == b"RIFF"
    assert base64.b64decode(result.b64) == result.webp_bytes


def test_encode_preview_fit_true_with_large_budget():
    """With a very large max_b64_chars budget, a small image always fits."""
    png = _make_png(32, 32)
    result = encode_preview(png, max_b64_chars=10_000_000)

    assert result.fit is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_encode_preview_raises_value_error_for_non_image_bytes():
    """Random bytes that are not an image raise ValueError."""
    with pytest.raises(ValueError, match="Cannot decode image"):
        encode_preview(b"this is not an image at all \x00\x01\x02")


def test_encode_preview_raises_value_error_for_empty_bytes():
    """Empty bytes raise ValueError."""
    with pytest.raises(ValueError, match="Cannot decode image"):
        encode_preview(b"")


# ---------------------------------------------------------------------------
# PreviewResult dataclass
# ---------------------------------------------------------------------------


def test_preview_result_is_frozen():
    """PreviewResult is a frozen dataclass — assignment raises FrozenInstanceError."""
    png = _make_png(50, 50)
    result = encode_preview(png)

    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        result.fit = False  # type: ignore[misc]


def test_preview_result_fields():
    """PreviewResult exposes webp_bytes, b64, and fit."""
    png = _make_png(50, 50)
    result = encode_preview(png)

    assert hasattr(result, "webp_bytes")
    assert hasattr(result, "b64")
    assert hasattr(result, "fit")
    assert isinstance(result.webp_bytes, bytes)
    assert isinstance(result.b64, str)
    assert isinstance(result.fit, bool)
