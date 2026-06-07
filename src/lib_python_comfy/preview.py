"""Inline preview encoding — convert raw image bytes to a WebP base64 preview.

The single public entry point :func:`encode_preview` opens any PIL-readable
image, optionally downscales it, and iteratively reduces WebP quality until
the base-64 representation fits within *max_b64_chars*.

All helpers are pure-mechanism; any policy (max dimension, budget, quality)
is caller-supplied.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class PreviewResult:
    """Result of an :func:`encode_preview` call.

    Attributes:
        webp_bytes: Raw WebP-encoded bytes.
        b64: Base-64 string representation of *webp_bytes*.
        fit: ``True`` when the encoded result fits within *max_b64_chars*;
            ``False`` when quality was floored at 1 and the budget was still
            exceeded.
    """

    webp_bytes: bytes
    b64: str
    fit: bool


def encode_preview(
    image_bytes: bytes,
    *,
    max_dim: int = 512,
    max_b64_chars: int = 100_000,
    quality: int = 70,
) -> PreviewResult:
    """Encode *image_bytes* as a downscaled WebP preview.

    Opens the image with PIL, downscales it so neither dimension exceeds
    *max_dim* (preserving aspect ratio), then iteratively reduces WebP quality
    by 5 per round until the base-64 representation is within *max_b64_chars*
    or quality reaches 1.

    Parameters
    ----------
    image_bytes:
        Raw bytes of any PIL-readable image format (PNG, JPEG, …).
    max_dim:
        Maximum width *and* height of the output image (default 512).
        Uses :meth:`~PIL.Image.Image.thumbnail` so the aspect ratio is
        preserved and the image is never upscaled.
    max_b64_chars:
        Maximum allowed length of the base-64 string (default 100 000).
    quality:
        Starting WebP quality (1–95, default 70).  Reduced by 5 each
        iteration; floored at 1.

    Returns:
        A :class:`PreviewResult` with ``fit=True`` when the budget was met,
        ``fit=False`` when quality=1 still exceeded *max_b64_chars*.

    Raises:
        ValueError: If *image_bytes* cannot be decoded as an image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    # Downscale in-place, preserving aspect ratio; no-op when already smaller.
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    # WebP requires RGB or RGBA; convert any exotic mode.
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    current_quality = max(1, min(quality, 95))

    while True:
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=current_quality)
        webp_bytes = buf.getvalue()
        b64 = base64.b64encode(webp_bytes).decode("ascii")

        if len(b64) <= max_b64_chars:
            return PreviewResult(webp_bytes=webp_bytes, b64=b64, fit=True)

        if current_quality <= 1:
            return PreviewResult(webp_bytes=webp_bytes, b64=b64, fit=False)

        current_quality = max(1, current_quality - 5)
