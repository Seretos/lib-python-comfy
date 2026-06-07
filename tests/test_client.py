"""Unit tests for ComfyClient.

All tests use ``httpx.MockTransport`` — no live ComfyUI server required.
The mock transport is injected via the optional ``transport=`` parameter on
``ComfyClient.__init__``, keeping module-level state untouched.
"""
from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from lib_python_comfy import ComfyClient, ComfyConnectionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    """Wrap a plain callable in an ``httpx.MockTransport``."""
    return httpx.MockTransport(handler)


def _json_response(data: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=data)


def _bytes_response(content: bytes, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, content=content)


def _error_transport() -> httpx.MockTransport:
    """Transport that always raises ``httpx.ConnectError``."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    return _mock_transport(handler)


def _timeout_transport() -> httpx.MockTransport:
    """Transport that always raises ``httpx.TimeoutException``."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    return _mock_transport(handler)


# ---------------------------------------------------------------------------
# queue_prompt
# ---------------------------------------------------------------------------


def test_queue_prompt_returns_prompt_id():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/prompt"
        body = json.loads(request.content)
        assert "prompt" in body
        return _json_response({"prompt_id": "abc", "number": 1})

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    result = client.queue_prompt({"node1": {}})
    assert result == "abc"


def test_queue_prompt_raises_on_transport_error():
    client = ComfyClient("http://localhost:8188", transport=_error_transport())
    with pytest.raises(ComfyConnectionError):
        client.queue_prompt({"node1": {}})


def test_queue_prompt_raises_on_timeout():
    client = ComfyClient("http://localhost:8188", transport=_timeout_transport())
    with pytest.raises(ComfyConnectionError):
        client.queue_prompt({"node1": {}})


def test_queue_prompt_raises_on_missing_prompt_id():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({})  # no prompt_id key

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    with pytest.raises(ValueError, match="prompt_id"):
        client.queue_prompt({"node1": {}})


def test_queue_prompt_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    with pytest.raises(ComfyConnectionError):
        client.queue_prompt({"node1": {}})


# ---------------------------------------------------------------------------
# get_history
# ---------------------------------------------------------------------------


def test_get_history_no_id():
    expected = {"123": {"outputs": {}}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/history"
        return _json_response(expected)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    assert client.get_history() == expected


def test_get_history_with_id():
    expected = {"outputs": {"node1": {}}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/history/abc-123"
        return _json_response(expected)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    assert client.get_history("abc-123") == expected


def test_get_history_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    with pytest.raises(ComfyConnectionError):
        client.get_history("missing-id")


# ---------------------------------------------------------------------------
# get_queue
# ---------------------------------------------------------------------------


def test_get_queue():
    expected = {"queue_running": [], "queue_pending": []}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/queue"
        return _json_response(expected)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    assert client.get_queue() == expected


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


def test_cancel():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/queue"
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    client.cancel("xyz")

    assert len(captured) == 1
    assert captured[0] == {"delete": ["xyz"]}


def test_cancel_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    with pytest.raises(ComfyConnectionError):
        client.cancel("xyz")


# ---------------------------------------------------------------------------
# get_object_info
# ---------------------------------------------------------------------------


def test_get_object_info_no_type():
    expected = {"KSampler": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info"
        return _json_response(expected)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    assert client.get_object_info() == expected


def test_get_object_info_with_type():
    expected = {"KSampler": {"input": {}}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/KSampler"
        return _json_response(expected)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    assert client.get_object_info("KSampler") == expected


# ---------------------------------------------------------------------------
# view_bytes
# ---------------------------------------------------------------------------


def test_view_bytes():
    raw = b"\x89PNG\r\n\x1a\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/view"
        # httpx stores params in request.url.params (already decoded)
        params = dict(request.url.params)
        assert params["filename"] == "my image.png"
        assert params["subfolder"] == "sub/folder"
        assert params["type"] == "output"
        return _bytes_response(raw)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    result = client.view_bytes(
        filename="my image.png", subfolder="sub/folder", type="output"
    )
    assert result == raw


def test_view_bytes_raises_on_transport_error():
    client = ComfyClient("http://localhost:8188", transport=_error_transport())
    with pytest.raises(ComfyConnectionError):
        client.view_bytes(filename="x.png", subfolder="", type="output")


def test_view_bytes_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    with pytest.raises(ComfyConnectionError):
        client.view_bytes(filename="missing.png", subfolder="", type="output")


# ---------------------------------------------------------------------------
# is_reachable
# ---------------------------------------------------------------------------


def test_is_reachable_true():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/system_stats"
        return httpx.Response(200)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    assert client.is_reachable() is True


def test_is_reachable_false_on_connect_error():
    client = ComfyClient("http://localhost:8188", transport=_error_transport())
    # Must return False, NOT raise ComfyConnectionError
    assert client.is_reachable() is False


def test_is_reachable_false_on_timeout():
    client = ComfyClient("http://localhost:8188", transport=_timeout_transport())
    assert client.is_reachable() is False


def test_is_reachable_false_on_non_200():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = ComfyClient("http://localhost:8188", transport=_mock_transport(handler))
    assert client.is_reachable() is False


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_closes_client():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"prompt_id": "ctx-1"})

    with ComfyClient("http://localhost:8188", transport=_mock_transport(handler)) as c:
        assert isinstance(c, ComfyClient)
        result = c.queue_prompt({"node": {}})
        assert result == "ctx-1"
    # After __exit__ the underlying client is closed; verify no exception was raised.


# ---------------------------------------------------------------------------
# ComfyConnectionError chaining
# ---------------------------------------------------------------------------


def test_connection_error_chains_original_cause():
    client = ComfyClient("http://localhost:8188", transport=_error_transport())
    with pytest.raises(ComfyConnectionError) as exc_info:
        client.queue_prompt({})
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)
