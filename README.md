# lib-python-comfy

ComfyUI API and workflow client library - programmatic access to ComfyUI graphs,
the prompt queue, and generated outputs.

A pure Python utility library: it supplies the *mechanism*; any *policy*
(names, paths, env-vars) is caller-supplied.

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

Dev / test install:

```bash
pip install -e ".[test]"
python -m pytest
```

With optional HuggingFace enrichment support:

```bash
pip install -e ".[hf]"
```

---

## Public API (35 names)

### Graph building

Build ComfyUI computation graphs programmatically.

```python
from lib_python_comfy import GraphBuilder, NodeRef, to_api, to_ui, txt2img, txt2audio, txt2video
```

**`GraphBuilder`** — incrementally build a ComfyUI graph.

```python
g = GraphBuilder()
node: NodeRef = g.add_node("SaveImage", {"filename_prefix": "ComfyUI"})
g.link(source_node, output_slot, target_node, "input_name")
```

**`NodeRef`** — opaque reference to a node inside a `GraphBuilder`. Has an `.id` attribute (int).

**`to_api(builder: GraphBuilder) -> dict`** — serialise to ComfyUI API format (same shape as `/prompt` expects).

**`to_ui(builder: GraphBuilder) -> dict`** — serialise to ComfyUI UI format (for loading into the web UI).

**`txt2img(*, model, positive, negative, width=512, height=512, steps=20, cfg=7.0, sampler_name="euler", scheduler="normal", seed=0) -> GraphBuilder`** — pre-wired text-to-image scaffold (7 nodes, fully linked):

```python
g = txt2img(
    model="v1-5-pruned-emaonly.safetensors",
    positive="a photo of a cat",
    negative="blurry, ugly",
    width=512,
    height=512,
    steps=20,
    cfg=7.0,
    sampler_name="euler",
    scheduler="normal",
    seed=42,
)
prompt = to_api(g)
```

**`txt2audio(*, model, positive, negative="", seconds=47.0, batch_size=1, sample_rate=44100, steps=20, cfg=3.5, sampler_name="dpmpp_3m_sde_gpu", scheduler="exponential", seed=0) -> GraphBuilder`** — pre-wired Stable Audio graph (7 nodes, fully linked):

```python
g = txt2audio(
    model="stable_audio_open_1.0.safetensors",
    positive="cinematic ambient music",
    negative="noise, distortion",
    seconds=47.0,
    steps=20,
    cfg=3.5,
    seed=42,
)
prompt = to_api(g)
```

**`txt2video(*, positive="", negative="", width=512, height=512, seed=0) -> GraphBuilder`** — always raises `NotImplementedError`; text-to-video generation requires a custom node (e.g. `VHS_VideoCombine` from ComfyUI-VideoHelper-Suite) that is not present in a stock ComfyUI install. Build the graph directly with `GraphBuilder` after installing a compatible video node.

---

### Running workflows

Submit graphs to a live ComfyUI server and poll for results.

```python
from lib_python_comfy import FlowRunner, RunResult, JobStatus, JobState
```

**`FlowRunner`** — high-level runner that submits a prompt and waits for completion.

**`RunResult`** — result of a completed run (contains `outputs`, `prompt_id`, etc.).

**`JobStatus`** — current status of a queued job.

**`JobState`** — enum of possible job states (`PENDING`, `RUNNING`, `DONE`, `FAILED`, ...).

---

### Assets

Parse, fetch, publish, and save ComfyUI output files.

```python
from lib_python_comfy import Asset, extract_assets, fetch_bytes, save_to_path, view_url, publish_asset
```

**`Asset`** — frozen Pydantic model representing one output file. Fields: `filename`, `subfolder`, `folder_type` (required); `mime_type`, `width`, `height`, `bytes_size` (optional, default `None`).

```python
a = Asset(filename="ComfyUI_00001_.png", subfolder="", folder_type="output")
a2 = Asset(filename="out.png", subfolder="", folder_type="output",
           mime_type="image/png", width=512, height=768, bytes_size=32768)
```

Identity is the triple `(filename, subfolder, folder_type)`; optional metadata fields do not affect `==` or `hash()`.

**`extract_assets(outputs: dict) -> list[Asset]`** — parse `history[prompt_id]["outputs"]` into Asset objects. Processes `"images"`, `"audio"`, and `"video"` output keys; other keys are silently ignored.

```python
assets = extract_assets(history["outputs"])
```

**`fetch_bytes(asset: Asset, client: ComfyClient) -> bytes`** — download raw bytes for an asset.

**`save_to_path(asset: Asset, dest: str | Path, client: ComfyClient) -> Path`** — download and write an asset; returns the resolved absolute path.

**`view_url(asset: Asset, base_url: str) -> str`** — construct the `/view` URL for an asset.

```python
url = view_url(asset, "http://localhost:8188")
# -> "http://localhost:8188/view?filename=ComfyUI_00001_.png&subfolder=&type=output"
```

**`publish_asset(asset: Asset, asset_bytes: bytes, dest_dir: str | Path, *, web_optimize: bool = False, max_bytes: int = 2_097_152) -> Path`** — write caller-supplied bytes to a local directory.

```python
path = publish_asset(asset, raw_bytes, "/srv/outputs")

# With WebP re-encoding (falls back to raw on non-image or oversized result):
path = publish_asset(asset, raw_bytes, "/srv/outputs", web_optimize=True, max_bytes=1_048_576)
```

- Any directory component in `asset.filename` is stripped (path-traversal protection).
- Raises `ValueError` if the resolved destination escapes `dest_dir`.

---

### Preview encoding

Convert raw image bytes to an inline WebP preview suitable for embedding.

```python
from lib_python_comfy import encode_preview, PreviewResult
```

**`encode_preview(image_bytes: bytes, *, max_dim: int = 512, max_b64_chars: int = 100_000, quality: int = 70) -> PreviewResult`**

```python
result = encode_preview(png_bytes)
result.webp_bytes  # raw WebP bytes
result.b64         # base-64 ASCII string
result.fit         # True if within max_b64_chars budget
```

- Downscales to `max_dim x max_dim` (aspect-ratio preserving, never upscales).
- Iteratively reduces quality by 5 per step until the base-64 string fits the budget or quality reaches 1.
- Raises `ValueError` on undecodable input.

**`PreviewResult`** — frozen dataclass: `webp_bytes: bytes`, `b64: str`, `fit: bool`.

---

### Workflow templates

Render ComfyUI API-format prompts from parameterised JSON templates.

```python
from lib_python_comfy import (
    render,
    load_builtin_template,
    list_builtin_templates,
    discover_params,
    TemplateParam,
    MissingParameterError,
)
```

**`list_builtin_templates() -> list[str]`** — return stem names of all built-in templates (e.g. `["txt2img_basic"]`).

**`load_builtin_template(name: str) -> dict`** — load a built-in template by stem name. Raises `FileNotFoundError` for unknown names.

**`discover_params(template: dict) -> list[TemplateParam]`** — discover all `PARAM_*` placeholders in a template.

**`render(template: dict, params: dict) -> dict`** — substitute placeholders with caller-supplied values.

```python
template = load_builtin_template("txt2img_basic")
params = discover_params(template)
# [TemplateParam(name='POSITIVE_PROMPT', type='STR', required=True), ...]

prompt = render(template, {"POSITIVE_PROMPT": "a cat", "STEPS": 20})
```

Placeholder syntax: `PARAM_<TYPE>_<NAME>` (required) or `PARAM_OPT_<TYPE>_<NAME>|default:<value>|` (optional).
Supported types: `STR`, `INT`, `FLOAT`, `BOOL`, `SEED`. `SEED` placeholders are always optional and auto-randomised.

**`TemplateParam`** — dataclass: `name: str`, `type: str`, `required: bool`.

**`MissingParameterError`** — raised by `render()` when a required parameter is absent.

---

### Node / model discovery

Query a live ComfyUI server for available resources.

```python
from lib_python_comfy import list_checkpoints, list_samplers, list_schedulers, list_node_types, get_node_schema, NodeTypeNotFoundError
```

**`list_checkpoints(client: ComfyClient) -> list[str]`** — available checkpoint model names.

**`list_samplers(client: ComfyClient) -> list[str]`** — available sampler names.

**`list_schedulers(client: ComfyClient) -> list[str]`** — available scheduler names.

**`list_node_types(client: ComfyClient) -> list[str]`** — all registered node class types.

**`get_node_schema(client: ComfyClient, node_type: str) -> dict`** — input schema for a specific node type. Returns `{"required": {...}, "optional": {...}}`. Raises `NodeTypeNotFoundError` when *node_type* is not present as a top-level key in the ComfyUI response.

**`NodeTypeNotFoundError`** — raised by `get_node_schema` when the requested node type is unknown. Has a `.node_type` attribute holding the queried name.

---

### Client

```python
from lib_python_comfy import ComfyClient, ComfyConnectionError
```

**`ComfyClient(base_url: str, *, transport=None)`** — low-level HTTP client for a ComfyUI server.

```python
client = ComfyClient("http://127.0.0.1:8188")

if not client.is_reachable():
    raise SystemExit("ComfyUI is not running")

prompt_id = client.queue_prompt(to_api(g))
history = client.get_history(prompt_id)
```

Use as a context manager to ensure the connection is closed:

```python
with ComfyClient("http://127.0.0.1:8188") as client:
    prompt_id = client.queue_prompt(my_workflow)
```

**`client.cancel(prompt_id: str) -> bool`** — remove *prompt_id* from the pending queue. Returns `True` if the prompt was found in `queue_pending` at the time of the call; `False` if it was not in the pending queue (running, already completed, or unknown). Raises `ComfyConnectionError` on transport failures.

**`ComfyConnectionError`** — raised when a connection to the server cannot be established.

---

### Serialisation guard

```python
from lib_python_comfy import SerializationGuard
```

**`SerializationGuard`** — context manager that prevents unsafe serialisation of graph objects outside a controlled scope.

---

## Off-surface helpers (not in `__all__`)

**`lib_python_comfy.hf_enrich.enrich_checkpoint(name: str, *, sha256: str | None = None) -> dict`** — optional HuggingFace model-card lookup. Requires `pip install -e ".[hf]"`. Never raises; returns `{}` when the dependency is missing or on network errors. Results are cached with `lru_cache(maxsize=256)`.

```python
from lib_python_comfy.hf_enrich import enrich_checkpoint

info = enrich_checkpoint("runwayml/stable-diffusion-v1-5")
# {"repo_id": "runwayml/stable-diffusion-v1-5", "tags": [...], "summary": "...", "url": "https://huggingface.co/..."}
```

---

## Versioning

Semantic versioning. The `version` in `pyproject.toml` is a placeholder on `main`;
`release.yml` stamps it onto the `release/Nx` branch and the `vX.Y.Z` tag.
Never hand-bump it.
