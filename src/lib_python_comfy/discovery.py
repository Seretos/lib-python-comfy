"""ComfyUI node/model discovery helpers.

Thin introspection layer on top of ``ComfyClient.get_object_info()``.
Every function accepts a :class:`~lib_python_comfy.client.ComfyClient`
and performs a single network call — there is no caching.

Transport errors (``ComfyConnectionError``) propagate unchanged; the
discovery layer adds no try/except around the client calls.
"""
from __future__ import annotations

from lib_python_comfy.client import ComfyClient


def list_checkpoints(client: ComfyClient) -> list[str]:
    """Return the list of available checkpoint model names.

    Reads ``CheckpointLoaderSimple.input.required.ckpt_name[0]`` from
    the full ``/object_info`` response.  Returns ``[]`` when the node or
    any intermediate key is absent, or when the extracted value is not a
    list.
    """
    response = client.get_object_info()
    node = response.get("CheckpointLoaderSimple", {})
    required = node.get("input", {}).get("required", {})
    value = required.get("ckpt_name", [None])[0]
    if not isinstance(value, list):
        return []
    return value


def list_samplers(client: ComfyClient) -> list[str]:
    """Return the list of available sampler names.

    Reads ``KSampler.input.required.sampler_name[0]`` from the full
    ``/object_info`` response.  Returns ``[]`` when the node or any
    intermediate key is absent, or when the extracted value is not a list.
    """
    response = client.get_object_info()
    node = response.get("KSampler", {})
    required = node.get("input", {}).get("required", {})
    value = required.get("sampler_name", [None])[0]
    if not isinstance(value, list):
        return []
    return value


def list_schedulers(client: ComfyClient) -> list[str]:
    """Return the list of available scheduler names.

    Reads ``KSampler.input.required.scheduler[0]`` from the full
    ``/object_info`` response.  Returns ``[]`` when the node or any
    intermediate key is absent, or when the extracted value is not a list.
    """
    response = client.get_object_info()
    node = response.get("KSampler", {})
    required = node.get("input", {}).get("required", {})
    value = required.get("scheduler", [None])[0]
    if not isinstance(value, list):
        return []
    return value


def list_node_types(client: ComfyClient) -> list[str]:
    """Return the list of all registered node type names.

    Returns the top-level keys of the ``/object_info`` response.
    Returns ``[]`` when the response is empty.
    """
    response = client.get_object_info()
    return list(response.keys())


def get_node_schema(client: ComfyClient, node_type: str) -> dict:
    """Return the input schema for a single node type.

    Calls ``GET /object_info/{node_type}`` and extracts
    ``input.required`` and ``input.optional``.  Both sub-dicts default to
    ``{}`` when the node is absent, ``input`` is missing, or either
    sub-key is absent.  Raw per-input values are passed through unchanged.

    Returns
    -------
    dict
        ``{"required": {...}, "optional": {...}}``
    """
    response = client.get_object_info(node_type)
    node = response.get(node_type, {})
    input_section = node.get("input", {})
    required = input_section.get("required", {})
    optional = input_section.get("optional", {})
    return {"required": required, "optional": optional}
