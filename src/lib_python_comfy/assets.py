"""Asset retrieval helpers.

Four public helpers covering the full lifecycle of a ComfyUI output asset:

- ``view_url``      — construct the ``/view`` URL for an asset.
- ``extract_assets``— parse the ``outputs`` dict from ``get_history`` into
                      a list of :class:`~lib_python_comfy.models.Asset` objects.
- ``fetch_bytes``   — download the raw bytes for an asset via a
                      :class:`~lib_python_comfy.client.ComfyClient`.
- ``save_to_path``  — download and write an asset to a local path.

All helpers are pure-mechanism; any policy (base URL, destination directory,
filename conventions) is caller-supplied.
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lib_python_comfy.models.asset import Asset

if TYPE_CHECKING:
    from lib_python_comfy.client import ComfyClient


def view_url(asset: Asset, base_url: str) -> str:
    """Return the absolute ``/view`` URL for *asset*.

    Query parameters are percent-encoded with ``%20`` for spaces (not ``+``),
    matching the ComfyUI upstream convention.

    Parameters
    ----------
    asset:
        The asset whose URL to build.
    base_url:
        Root URL of the ComfyUI server (trailing slash is stripped).
    """
    base = base_url.rstrip("/")
    # urllib.parse.quote encodes spaces as %20; safe='' encodes everything
    # except unreserved characters.
    params = urllib.parse.urlencode(
        {
            "filename": asset.filename,
            "subfolder": asset.subfolder,
            "type": asset.folder_type,
        },
        quote_via=urllib.parse.quote,
    )
    return f"{base}/view?{params}"


_MEDIA_KEYS = ("images", "audio", "video")


def extract_assets(outputs: dict[str, Any]) -> list[Asset]:
    """Parse a ComfyUI history ``outputs`` dict into a list of :class:`Asset`.

    *outputs* is the value of ``history[prompt_id]["outputs"]`` — a mapping
    from node ID to that node's output dictionary.  The keys ``"images"``,
    ``"audio"``, and ``"video"`` are all processed; any other keys are
    silently ignored.  Malformed or missing values are skipped defensively,
    as history dicts arrive from an external source.

    Parameters
    ----------
    outputs:
        The ``outputs`` sub-dict from a ``get_history`` response.
    """
    assets: list[Asset] = []
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for key in _MEDIA_KEYS:
            entries = node_output.get(key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    assets.append(
                        Asset(
                            filename=entry["filename"],
                            subfolder=entry["subfolder"],
                            folder_type=entry["type"],
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    # Missing keys or wrong types — skip gracefully.
                    continue
    return assets


def fetch_bytes(asset: Asset, client: ComfyClient) -> bytes:
    """Download and return the raw bytes for *asset*.

    Delegates to :meth:`~lib_python_comfy.client.ComfyClient.view_bytes`.

    Parameters
    ----------
    asset:
        The asset to fetch.
    client:
        An open :class:`~lib_python_comfy.client.ComfyClient`.
    """
    return client.view_bytes(asset.filename, asset.subfolder, asset.folder_type)


def save_to_path(asset: Asset, dest: str | Path, client: ComfyClient) -> Path:
    """Download *asset* and write it to *dest*, creating parent dirs as needed.

    Returns the resolved absolute :class:`~pathlib.Path` of the written file.

    Parameters
    ----------
    asset:
        The asset to fetch and save.
    dest:
        Destination file path (any string or :class:`~pathlib.Path`).
        Parent directories are created with ``mkdir(parents=True,
        exist_ok=True)``.
    client:
        An open :class:`~lib_python_comfy.client.ComfyClient`.
    """
    data = fetch_bytes(asset, client)
    resolved = Path(dest).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(data)
    return resolved
