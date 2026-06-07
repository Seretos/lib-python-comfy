"""Unit tests for Asset model and retrieval helpers.

All network-dependent tests use ``httpx.MockTransport`` — no live ComfyUI
server required.  The test structure mirrors ``test_client.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import httpx
import pytest

from lib_python_comfy import (
    Asset,
    ComfyClient,
    extract_assets,
    fetch_bytes,
    save_to_path,
    view_url,
)


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_client.py)
# ---------------------------------------------------------------------------


def _mock_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _bytes_response(content: bytes, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, content=content)


# ---------------------------------------------------------------------------
# Public API import smoke test
# ---------------------------------------------------------------------------


def test_public_api_importable():
    """Importing all five names from the top-level package must succeed."""
    from lib_python_comfy import Asset, extract_assets, fetch_bytes, save_to_path, view_url  # noqa: F401


# ---------------------------------------------------------------------------
# Asset model
# ---------------------------------------------------------------------------


def test_asset_required_fields_raise_validation_error():
    """Omitting any required field raises a Pydantic ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Asset(subfolder="", folder_type="output")  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        Asset(filename="x.png", folder_type="output")  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        Asset(filename="x.png", subfolder="")  # type: ignore[call-arg]


def test_asset_optional_metadata_defaults_none():
    """Optional metadata fields all default to None."""
    a = Asset(filename="out.png", subfolder="", folder_type="output")
    assert a.mime_type is None
    assert a.width is None
    assert a.height is None
    assert a.bytes_size is None


def test_asset_is_hashable():
    """Frozen model must be hashable and usable in sets."""
    a1 = Asset(filename="out.png", subfolder="", folder_type="output")
    a2 = Asset(filename="out.png", subfolder="", folder_type="output")
    a3 = Asset(filename="other.png", subfolder="", folder_type="output")

    assert hash(a1) == hash(a2)
    s = {a1, a2, a3}
    assert len(s) == 2
    assert a1 in s


def test_asset_identity_only_equality_and_hash():
    """Same identity triple but different metadata must compare equal and hash identically.

    This is the regression test for the blocking issue: Pydantic v2 frozen
    models include ALL fields in __eq__/__hash__ by default.  Without the
    explicit overrides an enriched Asset would be unequal to its plain
    counterpart, silently producing duplicates in sets/dicts.
    """
    a_plain = Asset(filename="out.png", subfolder="", folder_type="output")
    a_enriched = Asset(
        filename="out.png",
        subfolder="",
        folder_type="output",
        width=512,
        mime_type="image/png",
    )
    assert a_plain == a_enriched
    assert hash(a_plain) == hash(a_enriched)
    assert len({a_plain, a_enriched}) == 1


def test_asset_identity_fields_stored():
    """Fields are stored exactly as supplied."""
    a = Asset(filename="ComfyUI_00001_.png", subfolder="sub/dir", folder_type="temp")
    assert a.filename == "ComfyUI_00001_.png"
    assert a.subfolder == "sub/dir"
    assert a.folder_type == "temp"


def test_asset_optional_metadata_round_trip():
    """Optional metadata can be set and retrieved."""
    a = Asset(
        filename="out.png",
        subfolder="",
        folder_type="output",
        mime_type="image/png",
        width=512,
        height=768,
        bytes_size=32768,
    )
    assert a.mime_type == "image/png"
    assert a.width == 512
    assert a.height == 768
    assert a.bytes_size == 32768


# ---------------------------------------------------------------------------
# view_url
# ---------------------------------------------------------------------------


def test_view_url_basic():
    """Basic URL construction with no special characters."""
    a = Asset(filename="out.png", subfolder="", folder_type="output")
    url = view_url(a, "http://localhost:8188")
    assert url.startswith("http://localhost:8188/view?")
    assert "filename=out.png" in url
    assert "subfolder=" in url
    assert "type=output" in url


def test_view_url_encodes_spaces_as_percent20():
    """Spaces in filename must be encoded as ``%20``, not ``+``."""
    a = Asset(filename="my image.png", subfolder="", folder_type="output")
    url = view_url(a, "http://localhost:8188")
    assert "my%20image.png" in url
    assert "+" not in url


def test_view_url_encodes_subfolder_slashes():
    """Slashes in subfolder are encoded so the URL remains valid."""
    a = Asset(filename="out.png", subfolder="sub/folder", folder_type="output")
    url = view_url(a, "http://localhost:8188")
    # subfolder value should be present in encoded form
    assert "sub%2Ffolder" in url


def test_view_url_strips_trailing_slash():
    """A trailing slash on base_url and no slash produce identical URLs."""
    a = Asset(filename="out.png", subfolder="", folder_type="output")
    url_no_slash = view_url(a, "http://localhost:8188")
    url_with_slash = view_url(a, "http://localhost:8188/")
    assert url_no_slash == url_with_slash


def test_view_url_path_is_view():
    """The URL path component is always ``/view``."""
    a = Asset(filename="out.png", subfolder="", folder_type="output")
    url = view_url(a, "http://localhost:8188")
    parsed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(url)
    assert parsed.path == "/view"


# ---------------------------------------------------------------------------
# extract_assets
# ---------------------------------------------------------------------------

_REALISTIC_OUTPUTS = {
    "9": {
        "images": [
            {"filename": "out.png", "subfolder": "", "type": "output"},
        ]
    }
}


def test_extract_assets_regression_single_image():
    """Regression: realistic single-image history outputs -> one Asset."""
    assets = extract_assets(_REALISTIC_OUTPUTS)
    assert len(assets) == 1
    a = assets[0]
    assert a.filename == "out.png"
    assert a.subfolder == ""
    assert a.folder_type == "output"


def test_extract_assets_empty_outputs():
    """Empty dict produces an empty list."""
    assert extract_assets({}) == []


def test_extract_assets_multiple_nodes_and_images():
    """Multiple nodes and multiple images per node are all collected."""
    outputs = {
        "1": {
            "images": [
                {"filename": "a.png", "subfolder": "", "type": "output"},
                {"filename": "b.png", "subfolder": "", "type": "output"},
            ]
        },
        "2": {
            "images": [
                {"filename": "c.png", "subfolder": "sub", "type": "temp"},
            ]
        },
    }
    assets = extract_assets(outputs)
    assert len(assets) == 3
    filenames = {a.filename for a in assets}
    assert filenames == {"a.png", "b.png", "c.png"}


def test_extract_assets_skips_node_without_images_key():
    """A node whose output dict has no 'images' key is silently skipped."""
    outputs = {
        "1": {"latent": [{"filename": "latent.pt", "subfolder": "", "type": "output"}]},
        "2": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]},
    }
    assets = extract_assets(outputs)
    assert len(assets) == 1
    assert assets[0].filename == "out.png"


def test_extract_assets_audio_key_extracted_alongside_images():
    """'audio' key alongside 'images' are both extracted (regression: audio was previously ignored)."""
    outputs = {
        "5": {
            "audio": [{"filename": "out.wav", "subfolder": "", "type": "output"}],
            "images": [{"filename": "preview.png", "subfolder": "", "type": "temp"}],
        }
    }
    assets = extract_assets(outputs)
    assert len(assets) == 2
    filenames = {a.filename for a in assets}
    assert "out.wav" in filenames
    assert "preview.png" in filenames


def test_extract_assets_video_key_extracted():
    """'video' key entries are extracted into Asset objects."""
    outputs = {
        "6": {
            "video": [{"filename": "out.mp4", "subfolder": "", "type": "output"}],
        }
    }
    assets = extract_assets(outputs)
    assert len(assets) == 1
    assert assets[0].filename == "out.mp4"


def test_extract_assets_all_three_media_keys():
    """images, audio, and video entries from the same node are all collected."""
    outputs = {
        "7": {
            "images": [{"filename": "preview.png", "subfolder": "", "type": "temp"}],
            "audio": [{"filename": "out.wav", "subfolder": "", "type": "output"}],
            "video": [{"filename": "clip.mp4", "subfolder": "", "type": "output"}],
        }
    }
    assets = extract_assets(outputs)
    assert len(assets) == 3
    filenames = {a.filename for a in assets}
    assert filenames == {"preview.png", "out.wav", "clip.mp4"}


def test_extract_assets_unknown_key_still_ignored():
    """An unrecognised output key (e.g. 'latent') yields 0 assets."""
    outputs = {
        "8": {
            "latent": [{"filename": "latent.pt", "subfolder": "", "type": "output"}],
        }
    }
    assets = extract_assets(outputs)
    assert assets == []


def test_extract_assets_malformed_node_skipped():
    """Non-dict node outputs are skipped without raising."""
    outputs = {"1": None, "2": "unexpected string", "3": {"images": []}}
    assets = extract_assets(outputs)
    assert assets == []


def test_extract_assets_malformed_image_entry_skipped():
    """Image entries missing required keys are skipped without raising."""
    outputs = {
        "1": {
            "images": [
                {"filename": "good.png", "subfolder": "", "type": "output"},
                {"filename": "no_type.png", "subfolder": ""},  # missing "type"
                "not_a_dict",
            ]
        }
    }
    assets = extract_assets(outputs)
    assert len(assets) == 1
    assert assets[0].filename == "good.png"


# ---------------------------------------------------------------------------
# fetch_bytes
# ---------------------------------------------------------------------------


def test_fetch_bytes_delegates_to_view_bytes():
    """fetch_bytes calls /view with the correct query params and returns bytes."""
    raw = b"\x89PNG\r\n\x1a\n"
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _bytes_response(raw)

    asset = Asset(filename="out.png", subfolder="sub", folder_type="output")
    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    result = fetch_bytes(asset, client)

    assert result == raw
    assert len(captured) == 1
    req = captured[0]
    assert req.url.path == "/view"
    params = dict(req.url.params)
    assert params["filename"] == "out.png"
    assert params["subfolder"] == "sub"
    assert params["type"] == "output"


def test_fetch_bytes_encodes_spaces():
    """fetch_bytes correctly handles filenames with spaces (no assertion on
    encoding here — that is httpx's job — but the bytes are returned)."""
    raw = b"data"

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        assert params["filename"] == "my file.png"
        return _bytes_response(raw)

    asset = Asset(filename="my file.png", subfolder="", folder_type="output")
    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    assert fetch_bytes(asset, client) == raw


# ---------------------------------------------------------------------------
# save_to_path
# ---------------------------------------------------------------------------


def test_save_to_path_writes_bytes_and_returns_resolved_path(tmp_path: Path):
    """save_to_path writes the fetched bytes and returns a resolved Path."""
    raw = b"\x89PNG\r\n\x1a\nsome image data"

    def handler(request: httpx.Request) -> httpx.Response:
        return _bytes_response(raw)

    asset = Asset(filename="out.png", subfolder="", folder_type="output")
    dest = tmp_path / "out.png"
    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))

    result = save_to_path(asset, dest, client)

    assert result == dest.resolve()
    assert result.read_bytes() == raw


def test_save_to_path_creates_parent_directories(tmp_path: Path):
    """save_to_path creates missing intermediate directories."""
    raw = b"bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        return _bytes_response(raw)

    asset = Asset(filename="out.png", subfolder="", folder_type="output")
    # Nested path that doesn't exist yet
    dest = tmp_path / "deep" / "nested" / "dir" / "out.png"
    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))

    result = save_to_path(asset, dest, client)

    assert result.exists()
    assert result.read_bytes() == raw


def test_save_to_path_accepts_string_dest(tmp_path: Path):
    """save_to_path accepts a plain string as dest, not only Path."""
    raw = b"str dest"

    def handler(request: httpx.Request) -> httpx.Response:
        return _bytes_response(raw)

    asset = Asset(filename="out.png", subfolder="", folder_type="output")
    dest_str = str(tmp_path / "result.png")
    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))

    result = save_to_path(asset, dest_str, client)

    assert isinstance(result, Path)
    assert result.read_bytes() == raw
