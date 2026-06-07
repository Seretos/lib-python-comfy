"""Tests for the file-system asset publisher (ticket #15).

All tests use tmp_path — no network access required.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

import lib_python_comfy
from lib_python_comfy import Asset, publish_asset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(width: int = 64, height: int = 64) -> bytes:
    """Return raw bytes of an in-memory PNG."""
    img = Image.new("RGB", (width, height), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_asset(filename: str = "out.png") -> Asset:
    return Asset(filename=filename, subfolder="", folder_type="output")


# ---------------------------------------------------------------------------
# Public API smoke test
# ---------------------------------------------------------------------------


def test_publish_asset_in_public_all():
    """'publish_asset' must be in lib_python_comfy.__all__."""
    assert "publish_asset" in lib_python_comfy.__all__


def test_publish_asset_importable_from_top_level():
    """from lib_python_comfy import publish_asset succeeds."""
    from lib_python_comfy import publish_asset as pa  # noqa: F401
    assert pa is publish_asset


# ---------------------------------------------------------------------------
# Round-trip: bytes written, correct path returned
# ---------------------------------------------------------------------------


def test_publish_asset_round_trip(tmp_path: Path):
    """Published file exists, content matches, path is inside dest_dir."""
    raw = _make_png()
    asset = _make_asset("result.png")

    returned = publish_asset(asset, raw, tmp_path)

    assert returned.exists()
    assert returned.read_bytes() == raw
    assert returned.is_relative_to(tmp_path)


def test_publish_asset_returns_resolved_path(tmp_path: Path):
    """Returned path is absolute and resolved."""
    raw = b"data"
    asset = _make_asset("file.bin")

    returned = publish_asset(asset, raw, tmp_path)

    assert returned.is_absolute()
    assert returned == returned.resolve()


# ---------------------------------------------------------------------------
# Parent directory creation
# ---------------------------------------------------------------------------


def test_publish_asset_creates_parent_dirs(tmp_path: Path):
    """Nested dest_dir that does not yet exist is created automatically."""
    raw = b"bytes"
    asset = _make_asset("out.bin")
    nested = tmp_path / "a" / "b" / "c"

    returned = publish_asset(asset, raw, nested)

    assert returned.exists()
    assert returned.read_bytes() == raw


# ---------------------------------------------------------------------------
# Filename directory-component stripping
# ---------------------------------------------------------------------------


def test_publish_asset_strips_directory_component(tmp_path: Path):
    """A filename with a directory component is stripped to the bare name."""
    raw = b"safe"
    # asset.filename contains a path component that should be stripped
    asset = _make_asset("subdir/output.png")

    returned = publish_asset(asset, raw, tmp_path)

    # The written file must be directly inside tmp_path, not in a subdir.
    assert returned.parent == tmp_path.resolve()
    assert returned.name == "output.png"
    assert returned.read_bytes() == raw


# ---------------------------------------------------------------------------
# Path traversal rejection
# ---------------------------------------------------------------------------


def test_publish_asset_traversal_raises(tmp_path: Path, monkeypatch):
    """If the resolved path escapes dest_dir, ValueError is raised."""
    # We construct a case where Path(asset.filename).name is innocuous but
    # the resolved path still escapes — simulate by monkeypatching resolve.
    raw = b"bad"
    asset = _make_asset("innocent.png")

    outside = tmp_path.parent / "outside.png"

    original_resolve = Path.resolve

    def _fake_resolve(self: Path, strict: bool = False) -> Path:
        # Only intercept the dest candidate, not dest_root itself
        if self.name == "innocent.png":
            return outside
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _fake_resolve)

    with pytest.raises(ValueError, match="[Pp]ath traversal"):
        publish_asset(asset, raw, tmp_path)


# ---------------------------------------------------------------------------
# web_optimize=False (default) — bytes pass through unchanged
# ---------------------------------------------------------------------------


def test_publish_asset_no_web_optimize_bytes_identical(tmp_path: Path):
    """Without web_optimize the file content is exactly asset_bytes."""
    raw = _make_png()
    asset = _make_asset("out.png")

    returned = publish_asset(asset, raw, tmp_path, web_optimize=False)

    assert returned.read_bytes() == raw
    assert returned.suffix == ".png"


# ---------------------------------------------------------------------------
# web_optimize=True with a valid PNG — re-encoded as .webp
# ---------------------------------------------------------------------------


def test_publish_asset_web_optimize_produces_webp(tmp_path: Path):
    """web_optimize=True on a valid PNG writes a .webp file."""
    raw = _make_png()
    asset = _make_asset("out.png")

    returned = publish_asset(asset, raw, tmp_path, web_optimize=True)

    assert returned.suffix == ".webp"
    assert returned.exists()
    # WebP magic: starts with RIFF
    assert returned.read_bytes()[:4] == b"RIFF"


def test_publish_asset_web_optimize_max_bytes_exceeded_falls_back(tmp_path: Path):
    """When the WebP result exceeds max_bytes, fall back to raw bytes and original extension."""
    raw = _make_png(200, 200)
    asset = _make_asset("out.png")

    # max_bytes=1 forces WebP to always exceed the limit -> fallback
    returned = publish_asset(asset, raw, tmp_path, web_optimize=True, max_bytes=1)

    assert returned.suffix == ".png"
    assert returned.read_bytes() == raw


# ---------------------------------------------------------------------------
# web_optimize=True with non-image bytes — graceful fallback, no ValueError
# ---------------------------------------------------------------------------


def test_publish_asset_web_optimize_non_image_fallback(tmp_path: Path):
    """web_optimize=True on non-image bytes does NOT raise; falls back to raw."""
    raw = b"this is not an image \x00\x01\x02"
    asset = _make_asset("data.bin")

    # Must not raise
    returned = publish_asset(asset, raw, tmp_path, web_optimize=True)

    assert returned.exists()
    assert returned.read_bytes() == raw
    # Extension must be the original (no .webp)
    assert returned.suffix == ".bin"
