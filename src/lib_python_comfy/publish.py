"""File-system asset publisher.

The single public entry point :func:`publish_asset` writes caller-supplied
bytes for an :class:`~lib_python_comfy.models.Asset` to a local destination
directory, with optional WebP re-encoding via :func:`~lib_python_comfy.preview.encode_preview`.

All helpers are pure-mechanism; any policy (destination root, size limits,
optimisation flags) is caller-supplied.
"""
from __future__ import annotations

from pathlib import Path

from lib_python_comfy.models.asset import Asset
from lib_python_comfy.preview import encode_preview


def publish_asset(
    asset: Asset,
    asset_bytes: bytes,
    dest_dir: str | Path,
    *,
    web_optimize: bool = False,
    max_bytes: int = 2_097_152,
) -> Path:
    """Write *asset_bytes* to *dest_dir* and return the resolved destination path.

    Parameters
    ----------
    asset:
        The :class:`~lib_python_comfy.models.Asset` whose bytes are being
        published.  Only ``asset.filename`` is used (any directory component is
        stripped to prevent path traversal).
    asset_bytes:
        Raw bytes to write.  The caller is responsible for fetching them (e.g.
        via :func:`~lib_python_comfy.assets.fetch_bytes`).
    dest_dir:
        Root directory to publish into.  Parent directories are created with
        ``mkdir(parents=True, exist_ok=True)``.
    web_optimize:
        When ``True``, attempt to re-encode the asset as WebP via
        :func:`~lib_python_comfy.preview.encode_preview`.  If the result is
        within *max_bytes* the ``.webp`` extension is used; otherwise the
        original bytes and extension are kept.  Non-image assets fall back to
        raw bytes silently.
    max_bytes:
        Maximum file size (in bytes) for the WebP output to be accepted when
        *web_optimize* is ``True`` (default 2 MiB).  Has no effect when
        *web_optimize* is ``False``.

    Returns:
        The resolved absolute :class:`~pathlib.Path` of the written file.

    Raises:
        ValueError: If the resolved destination path escapes *dest_dir*
            (path-traversal attempt).
    """
    dest_root = Path(dest_dir).resolve()

    # Strip any directory component from the filename to prevent traversal.
    safe_name = Path(asset.filename).name
    dest = dest_root / safe_name

    # Guard against traversal (e.g. if safe_name somehow contains symlinks).
    if not dest.resolve().is_relative_to(dest_root):
        raise ValueError(
            f"Path traversal detected: {asset.filename!r} resolves outside {dest_dir!r}"
        )

    final_bytes = asset_bytes
    final_dest = dest

    if web_optimize:
        try:
            result = encode_preview(asset_bytes, max_dim=4096, max_b64_chars=10**18, quality=80)
            if len(result.webp_bytes) <= max_bytes:
                final_bytes = result.webp_bytes
                # Replace extension with .webp
                final_dest = dest.with_suffix(".webp")
        except ValueError:
            # Non-image asset — fall back to raw bytes with original extension.
            pass

    final_dest.parent.mkdir(parents=True, exist_ok=True)
    final_dest.write_bytes(final_bytes)
    return final_dest.resolve()
