"""ComfyUI HTTP client.

Transport-only wrapper around ComfyUI's REST API. All policy (base URL,
credentials, retry logic) is caller-supplied; this module only provides
the mechanism.
"""
from __future__ import annotations

from typing import Any

import httpx


class ComfyConnectionError(Exception):
    """Raised when an HTTP transport/network failure occurs talking to ComfyUI.

    The original ``httpx`` exception is always chained as ``__cause__`` so
    callers can inspect it without coupling their code to ``httpx`` directly.
    """


class ComfyClient:
    """Typed HTTP client for the ComfyUI REST API.

    Parameters
    ----------
    base_url:
        Root URL of the ComfyUI server, e.g. ``"http://127.0.0.1:8188"``.
        Trailing slash is optional — it is stripped internally.
    transport:
        Optional ``httpx.BaseTransport`` to inject (useful for unit tests
        via ``httpx.MockTransport``).  When ``None`` (the default) a real
        network transport is used.
    """

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        kwargs: dict[str, Any] = {}
        if transport is not None:
            kwargs["transport"] = transport
        self._client = httpx.Client(**kwargs)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> ComfyClient:
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        """Build an absolute URL from a relative *path* (must start with ``/``)."""
        return f"{self._base_url}{path}"

    def _get(self, path: str, **params: str) -> httpx.Response:
        url = self._url(path)
        try:
            resp = self._client.get(url, params=params or None)
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as err:
            raise ComfyConnectionError(str(err)) from err

    def _post(self, path: str, json: Any = None) -> httpx.Response:
        url = self._url(path)
        try:
            resp = self._client.post(url, json=json)
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as err:
            raise ComfyConnectionError(str(err)) from err

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_reachable(self) -> bool:
        """Return ``True`` when ComfyUI responds to ``GET /system_stats``.

        This is a best-effort health probe — it never raises; transport
        errors are silently converted to ``False``.
        """
        try:
            resp = self._client.get(self._url("/system_stats"))
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def queue_prompt(self, prompt: dict) -> str:
        """Enqueue a prompt and return the assigned ``prompt_id``.

        Raises
        ------
        ComfyConnectionError
            On any transport/network failure.
        ValueError
            When the server response does not contain a ``prompt_id`` key.
        """
        resp = self._post("/prompt", json={"prompt": prompt})
        data: dict = resp.json()
        if "prompt_id" not in data:
            raise ValueError(
                f"ComfyUI response missing 'prompt_id'; got keys: {list(data)}"
            )
        return data["prompt_id"]

    def get_history(self, prompt_id: str | None = None) -> dict:
        """Return the execution history.

        When *prompt_id* is given, returns the history for that single
        prompt (``GET /history/{prompt_id}``); otherwise returns the full
        history (``GET /history``).
        """
        path = f"/history/{prompt_id}" if prompt_id is not None else "/history"
        return self._get(path).json()

    def get_queue(self) -> dict:
        """Return the current queue status (``GET /queue``)."""
        return self._get("/queue").json()

    def cancel(self, prompt_id: str) -> None:
        """Remove *prompt_id* from the pending queue (``POST /queue``)."""
        self._post("/queue", json={"delete": [prompt_id]})

    def get_object_info(self, node_type: str | None = None) -> dict:
        """Return object/node metadata.

        When *node_type* is given, returns metadata for that node type
        only (``GET /object_info/{node_type}``); otherwise returns all
        nodes (``GET /object_info``).
        """
        path = (
            f"/object_info/{node_type}" if node_type is not None else "/object_info"
        )
        return self._get(path).json()

    def view_bytes(self, filename: str, subfolder: str, type: str) -> bytes:
        """Fetch a generated image/file and return its raw bytes.

        Issues ``GET /view?filename=...&subfolder=...&type=...`` with
        properly URL-encoded query parameters.
        """
        return self._get(
            "/view", filename=filename, subfolder=subfolder, type=type
        ).content
