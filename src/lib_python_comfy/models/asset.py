"""Asset model — first-class representation of a single ComfyUI output file.

The identity of an asset is the triple ``(filename, subfolder, folder_type)``.
The Python field is ``folder_type``; the ComfyUI wire parameter is ``type``.
That mismatch is bridged in ``assets.py``, not here.

The model is **frozen** so it is hashable and safe to store in sets/dicts.
"""
from __future__ import annotations

from pydantic import BaseModel


class Asset(BaseModel, frozen=True):
    """A single generated file produced by a ComfyUI execution.

    Parameters
    ----------
    filename:
        The bare filename as returned by ComfyUI (e.g. ``"ComfyUI_00001_.png"``).
    subfolder:
        Sub-directory within the output root, often an empty string.
    folder_type:
        ComfyUI folder type (e.g. ``"output"``, ``"input"``, ``"temp"``).
        Corresponds to the ``type`` query-parameter in ``/view``.
    mime_type:
        Optional MIME type hint (e.g. ``"image/png"``).
    width:
        Optional image width in pixels.
    height:
        Optional image height in pixels.
    bytes_size:
        Optional file size in bytes.
    """

    filename: str
    subfolder: str
    folder_type: str

    # Optional metadata — populated by higher-level helpers, not the core model.
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    bytes_size: int | None = None

    # Identity is the triple (filename, subfolder, folder_type) only.
    # Pydantic v2 frozen models include ALL fields in __eq__/__hash__ by
    # default, which would make an asset enriched with metadata compare
    # unequal to the same unenriched asset — contradicting the contract.
    # These explicit overrides restrict equality and hashing to the identity
    # triple so that callers can enrich assets freely without breaking
    # set/dict deduplication.

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Asset):
            return NotImplemented
        return (self.filename, self.subfolder, self.folder_type) == (
            other.filename, other.subfolder, other.folder_type
        )

    def __hash__(self) -> int:
        return hash((self.filename, self.subfolder, self.folder_type))
