"""lib-python-comfy — ComfyUI API and workflow client library - programmatic access to ComfyUI graphs, the prompt queue, and generated outputs

Public re-exports. Everything a consumer is meant to import lives here;
adding, removing, or changing the signature of a name in `__all__` is a
breaking change. Keep `__all__`, the README, and the version in sync.
See `README.md` for usage.
"""
from __future__ import annotations

from lib_python_comfy.assets import extract_assets, fetch_bytes, save_to_path, view_url
from lib_python_comfy.client import ComfyClient, ComfyConnectionError
from lib_python_comfy.discovery import (
    get_node_schema,
    list_checkpoints,
    list_node_types,
    list_samplers,
    list_schedulers,
)
from lib_python_comfy.graph import GraphBuilder, NodeRef, to_api, to_ui
from lib_python_comfy.models import Asset
from lib_python_comfy.serialization import SerializationGuard
from lib_python_comfy.templates import (
    MissingParameterError,
    TemplateParam,
    discover_params,
    list_builtin_templates,
    load_builtin_template,
    render,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Asset",
    "ComfyClient",
    "ComfyConnectionError",
    "SerializationGuard",
    "GraphBuilder",
    "NodeRef",
    "to_api",
    "to_ui",
    "MissingParameterError",
    "TemplateParam",
    "discover_params",
    "list_builtin_templates",
    "load_builtin_template",
    "render",
    "extract_assets",
    "fetch_bytes",
    "save_to_path",
    "view_url",
    "list_checkpoints",
    "list_samplers",
    "list_schedulers",
    "list_node_types",
    "get_node_schema",
]
