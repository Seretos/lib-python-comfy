"""Static node-type metadata used by :func:`lib_python_comfy.graph.ui_format.to_ui`.

This is a small, hand-maintained registry of the known scaffold node types
(see ``scaffolds.py``) that lets ``to_ui`` emit real ComfyUI slot types and
``widgets_values`` instead of the "UNKNOWN" sentinel.  Any node type not
present here safely falls back to the previous behaviour — this module must
never raise for an unrecognised ``class_type``.
"""
from __future__ import annotations

# Sentinel used by ``control_after_generate`` — a litegraph-only companion
# widget for the "seed" widget that has no backing value in raw_inputs.
_CONTROL_AFTER_GENERATE_DEFAULT = "fixed"

NODE_META: dict[str, dict] = {
    "CheckpointLoaderSimple": {
        "outputs": ["MODEL", "CLIP", "VAE"],
        "inputs": {},
        "widgets": ["ckpt_name"],
    },
    "CLIPTextEncode": {
        "outputs": ["CONDITIONING"],
        "inputs": {"clip": "CLIP"},
        "widgets": ["text"],
    },
    "EmptyLatentImage": {
        "outputs": ["LATENT"],
        "inputs": {},
        "widgets": ["width", "height", "batch_size"],
    },
    "EmptyLatentAudio": {
        "outputs": ["LATENT"],
        "inputs": {},
        "widgets": ["seconds", "batch_size"],
    },
    "KSampler": {
        "outputs": ["LATENT"],
        "inputs": {
            "model": "MODEL",
            "positive": "CONDITIONING",
            "negative": "CONDITIONING",
            "latent_image": "LATENT",
        },
        "widgets": [
            "seed",
            "control_after_generate",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "denoise",
        ],
    },
    "VAEDecode": {
        "outputs": ["IMAGE"],
        "inputs": {"samples": "LATENT", "vae": "VAE"},
        "widgets": [],
    },
    "VAEDecodeAudio": {
        "outputs": ["AUDIO"],
        "inputs": {"samples": "LATENT", "vae": "VAE"},
        "widgets": [],
    },
    "SaveImage": {
        "outputs": [],
        "inputs": {"images": "IMAGE"},
        "widgets": ["filename_prefix"],
    },
    "SaveAudio": {
        "outputs": [],
        "inputs": {"audio": "AUDIO"},
        "widgets": ["filename_prefix"],
    },
}


def output_slot(class_type: str, idx: int) -> tuple[str, str]:
    """Return ``(name, type)`` for output slot *idx* of *class_type*.

    Falls back to ``(f"OUTPUT_{idx}", "UNKNOWN")`` when *class_type* is not
    registered or *idx* is out of range for the registered outputs list.
    """
    meta = NODE_META.get(class_type)
    if meta is None:
        return (f"OUTPUT_{idx}", "UNKNOWN")
    outputs = meta["outputs"]
    if idx < 0 or idx >= len(outputs):
        return (f"OUTPUT_{idx}", "UNKNOWN")
    slot_type = outputs[idx]
    return (slot_type, slot_type)


def input_type(class_type: str, input_name: str) -> str:
    """Return the real ComfyUI type name for *input_name* on *class_type*.

    Falls back to ``"UNKNOWN"`` when *class_type* is not registered or
    *input_name* is not a known link-input for that type.
    """
    meta = NODE_META.get(class_type)
    if meta is None:
        return "UNKNOWN"
    return meta["inputs"].get(input_name, "UNKNOWN")


def widget_order(class_type: str) -> list[str] | None:
    """Return the ordered widget names for *class_type*, or ``None``.

    ``None`` means *class_type* is not registered (no ``widgets_values`` key
    should be emitted at all).  A registered type with no widgets returns an
    empty list — callers should also omit ``widgets_values`` in that case.
    """
    meta = NODE_META.get(class_type)
    if meta is None:
        return None
    return list(meta["widgets"])


def widget_value(raw_inputs: dict, widget_name: str):
    """Return the value to use for *widget_name* in ``widgets_values``.

    ``control_after_generate`` has no backing entry in ``raw_inputs`` (it is
    a litegraph companion widget for "seed" with no runtime value); it is
    always substituted with the literal default ``"fixed"``.
    """
    if widget_name == "control_after_generate":
        return _CONTROL_AFTER_GENERATE_DEFAULT
    return raw_inputs.get(widget_name)
