"""ComfyUI HTTP client.

Transport-only wrapper around ComfyUI's REST API. All policy (base URL,
credentials, retry logic) is caller-supplied; this module only provides
the mechanism.
"""
from __future__ import annotations

from typing import Any, Never

import httpx


class ComfyConnectionError(Exception):
    """Raised when an HTTP transport/network failure occurs talking to ComfyUI.

    The original ``httpx`` exception is always chained as ``__cause__`` so
    callers can inspect it without coupling their code to ``httpx`` directly.

    Attributes
    ----------
    detail:
        Parsed JSON body from the server response, or ``None`` when the
        response carried no JSON body (network errors, plain-text responses).
    """

    def __init__(self, message: str, *, detail: object = None) -> None:
        super().__init__(message)
        self.detail = detail


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

    def _raise_http_error(self, err: httpx.HTTPError) -> Never:
        """Normalize an ``httpx.HTTPError`` into a ``ComfyConnectionError``.

        For ``httpx.HTTPStatusError`` (raised by ``raise_for_status``):

        - strips the MDN boilerplate URL appended by httpx;
        - parses the response body as JSON and stores it in ``detail``;
        - for 404 responses that carry a ``filename`` query parameter,
          produces a terse ``"file not found: <filename>"`` message instead.

        For all other ``httpx.HTTPError`` subclasses (``ConnectError``,
        ``TimeoutException``, …) the message is passed through as-is with
        ``detail=None``.
        """
        if isinstance(err, httpx.HTTPStatusError):
            # Strip MDN boilerplate appended by httpx on a new line:
            # "… \nFor more information check: https://developer.mozilla.org/…"
            raw_msg = str(err)
            clean_msg = raw_msg.split("\nFor more information")[0].strip()

            detail: object = None
            try:
                detail = err.response.json()
            except Exception:
                pass

            # 404 with a filename param → terse "file not found: <name>" message
            if err.response.status_code == 404:
                params = dict(err.request.url.params)
                filename = params.get("filename")
                if filename:
                    clean_msg = f"file not found: {filename}"

            raise ComfyConnectionError(clean_msg, detail=detail) from err

        # Network-level errors (ConnectError, TimeoutException, …): no response body
        raise ComfyConnectionError(str(err)) from err

    def _get(self, path: str, **params: str) -> httpx.Response:
        url = self._url(path)
        try:
            resp = self._client.get(url, params=params or None)
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as err:
            self._raise_http_error(err)

    def _post(self, path: str, json: Any = None) -> httpx.Response:
        url = self._url(path)
        try:
            resp = self._client.post(url, json=json)
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as err:
            self._raise_http_error(err)

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

    def cancel(self, prompt_id: str) -> bool:
        """Remove *prompt_id* from the pending queue (``POST /queue``).

        Checks the live queue before posting the delete request so the return
        value reflects whether the prompt was actually pending at the time of
        the call.

        Returns
        -------
        bool
            ``True`` if *prompt_id* was found in ``queue_pending`` (and the
            delete request was sent); ``False`` if it was not in the pending
            queue — either because it is currently running, already completed,
            or the id is unknown.

        Raises
        ------
        ComfyConnectionError
            On any transport/network failure during either HTTP call.
        """
        queue = self.get_queue()
        # Queue entry shape: [number, prompt_id, prompt_dict, extra_data, outputs_to_execute]
        was_pending = any(
            entry[1] == prompt_id for entry in queue.get("queue_pending", [])
        )
        self._post("/queue", json={"delete": [prompt_id]})
        return was_pending

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
