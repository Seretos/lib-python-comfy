"""Unit tests for discovery helpers.

All tests use ``httpx.MockTransport`` — no live ComfyUI server required.
The mock transport is injected via the optional ``transport=`` parameter on
``ComfyClient.__init__``, keeping module-level state untouched.
"""
from __future__ import annotations

import httpx
import pytest

from lib_python_comfy import (
    ComfyClient,
    ComfyConnectionError,
    get_node_schema,
    list_checkpoints,
    list_node_types,
    list_samplers,
    list_schedulers,
)


# ---------------------------------------------------------------------------
# Shared fixture payload
# ---------------------------------------------------------------------------

OBJECT_INFO = {
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["v1-5.ckpt", "sd_xl.safetensors"], "MODEL"]}}
    },
    "KSampler": {
        "input": {
            "required": {
                "sampler_name": [["euler", "dpm_2"], "SAMPLER"],
                "scheduler":    [["normal", "karras"], "SCHEDULER"],
            },
            "optional": {"latent_image": ["LATENT"]},
        }
    },
    "CLIPTextEncode": {
        "input": {"required": {"text": ["STRING", {"multiline": True}]}, "optional": {}}
    },
}


# ---------------------------------------------------------------------------
# Helpers (mirrors test_client.py conventions)
# ---------------------------------------------------------------------------


def _json_response(data: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=data)


def _make_client(response_data: dict, expected_path: str | None = None) -> ComfyClient:
    """Build a ``ComfyClient`` backed by a mock that returns *response_data*."""

    def handler(request: httpx.Request) -> httpx.Response:
        if expected_path is not None:
            assert request.url.path == expected_path
        return _json_response(response_data)

    return ComfyClient("http://localhost:8188", transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# list_checkpoints
# ---------------------------------------------------------------------------


def test_list_checkpoints_happy_path():
    client = _make_client(OBJECT_INFO, "/object_info")
    result = list_checkpoints(client)
    assert result == ["v1-5.ckpt", "sd_xl.safetensors"]


def test_list_checkpoints_missing_node():
    """CheckpointLoaderSimple absent → []."""
    data = {k: v for k, v in OBJECT_INFO.items() if k != "CheckpointLoaderSimple"}
    client = _make_client(data)
    assert list_checkpoints(client) == []


def test_list_checkpoints_missing_ckpt_name_key():
    """ckpt_name key absent → []."""
    data = {
        "CheckpointLoaderSimple": {
            "input": {"required": {}}  # no ckpt_name
        }
    }
    client = _make_client(data)
    assert list_checkpoints(client) == []


def test_list_checkpoints_non_list_value():
    """ckpt_name[0] is a string, not a list → []."""
    data = {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": ["STRING", {}]}}
        }
    }
    client = _make_client(data)
    assert list_checkpoints(client) == []


# ---------------------------------------------------------------------------
# list_samplers
# ---------------------------------------------------------------------------


def test_list_samplers_happy_path():
    client = _make_client(OBJECT_INFO, "/object_info")
    assert list_samplers(client) == ["euler", "dpm_2"]


def test_list_samplers_missing_node():
    """KSampler absent → []."""
    data = {k: v for k, v in OBJECT_INFO.items() if k != "KSampler"}
    client = _make_client(data)
    assert list_samplers(client) == []


# ---------------------------------------------------------------------------
# list_schedulers
# ---------------------------------------------------------------------------


def test_list_schedulers_happy_path():
    client = _make_client(OBJECT_INFO, "/object_info")
    assert list_schedulers(client) == ["normal", "karras"]


def test_list_schedulers_missing_node():
    """KSampler absent → []."""
    data = {k: v for k, v in OBJECT_INFO.items() if k != "KSampler"}
    client = _make_client(data)
    assert list_schedulers(client) == []


# ---------------------------------------------------------------------------
# list_node_types
# ---------------------------------------------------------------------------


def test_list_node_types_happy_path():
    client = _make_client(OBJECT_INFO, "/object_info")
    result = list_node_types(client)
    assert set(result) == {"CheckpointLoaderSimple", "KSampler", "CLIPTextEncode"}


def test_list_node_types_empty_response():
    client = _make_client({})
    assert list_node_types(client) == []


# ---------------------------------------------------------------------------
# get_node_schema
# ---------------------------------------------------------------------------


def test_get_node_schema_happy_path():
    """Returns required and optional dicts for KSampler."""
    # get_object_info(node_type) hits /object_info/KSampler and still
    # returns a top-level dict keyed by node type.
    data = {"KSampler": OBJECT_INFO["KSampler"]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/KSampler"
        return _json_response(data)

    client = ComfyClient("http://localhost:8188", transport=httpx.MockTransport(handler))
    schema = get_node_schema(client, "KSampler")

    assert schema["required"] == OBJECT_INFO["KSampler"]["input"]["required"]
    assert schema["optional"] == OBJECT_INFO["KSampler"]["input"]["optional"]


def test_get_node_schema_missing_node():
    """Node absent from response → {"required": {}, "optional": {}}."""
    client = _make_client({})
    schema = get_node_schema(client, "NonExistentNode")
    assert schema == {"required": {}, "optional": {}}


def test_get_node_schema_no_input_key():
    """Node present but 'input' key absent → {"required": {}, "optional": {}}."""
    data = {"KSampler": {}}  # no "input" key
    client = _make_client(data)
    schema = get_node_schema(client, "KSampler")
    assert schema == {"required": {}, "optional": {}}


def test_get_node_schema_no_optional_key():
    """'optional' sub-key absent → optional defaults to {}."""
    data = {
        "KSampler": {
            "input": {
                "required": {"sampler_name": [["euler"], "SAMPLER"]},
                # no "optional" key
            }
        }
    }
    client = _make_client(data)
    schema = get_node_schema(client, "KSampler")
    assert schema["optional"] == {}
    assert schema["required"] == {"sampler_name": [["euler"], "SAMPLER"]}
