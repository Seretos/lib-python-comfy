# lib-python-comfy

ComfyUI API and workflow client library - programmatic access to ComfyUI graphs, the prompt queue, and generated outputs

A pure Python library — no binary, no MCP, no marketplace. Consumed as
source by downstream projects via a git pin.

## Install

Pin an exact tag (recommended) or the floating major-release branch:

```bash
# exact tag
pip install "git+https://github.com/Seretos/lib-python-comfy@v0.1.0"

# floating: latest 0.x.y release
pip install "git+https://github.com/Seretos/lib-python-comfy@release/0.x"
```

Or in a consumer's `pyproject.toml`:

```toml
dependencies = [
  "lib-python-comfy @ git+https://github.com/Seretos/lib-python-comfy@v0.1.0",
]
```

## Usage

```python
import lib_python_comfy

print(lib_python_comfy.__version__)
```

### ComfyClient

`ComfyClient` is a transport-only HTTP wrapper around the ComfyUI REST API.
Supply the base URL yourself — the client has no opinion on environment
variables or default ports.

```python
from lib_python_comfy import ComfyClient, ComfyConnectionError

client = ComfyClient("http://127.0.0.1:8188")

# Health check (never raises)
if not client.is_reachable():
    raise SystemExit("ComfyUI is not running")

# Queue a prompt and wait for results
try:
    prompt_id = client.queue_prompt({"node1": {"class_type": "KSampler", "inputs": {}}})
except ComfyConnectionError as exc:
    raise SystemExit(f"Network error: {exc}") from exc

history = client.get_history(prompt_id)

# Fetch a generated image
image_bytes = client.view_bytes(
    filename="ComfyUI_00001_.png",
    subfolder="",
    type="output",
)
```

Use it as a context manager to ensure the underlying HTTP connection is
closed when you are done:

```python
with ComfyClient("http://127.0.0.1:8188") as client:
    prompt_id = client.queue_prompt(my_workflow)
```

See `src/lib_python_comfy/__init__.py` for the full public API (`__all__`).

## Develop

```bash
pip install -e ".[test]"
python -m pytest
```

## Versioning

Semantic versioning. The `version` in `pyproject.toml` is a placeholder
on `main` — the release workflow stamps it onto the `release/Nx` branch
and the `vX.Y.Z` tag. Don't hand-bump it.
