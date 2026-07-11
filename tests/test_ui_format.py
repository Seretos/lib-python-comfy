"""Tests for scaffold-grounded to_ui() output (ticket #34).

Regression coverage for: to_ui() previously emitted "UNKNOWN" for every slot
type (inputs, outputs, and the link-type field), never emitted
widgets_values, and never emitted last_node_id/last_link_id — so a
ComfyUI-exported workflow opened with no drawn links and no runnable widget
state ("disconnected nodes"). These tests build real scaffold graphs and
assert real type names, widgets_values, and top-level id bookkeeping are
present; the generic-node fallback (unknown class_type) is exercised too,
to confirm it still degrades exactly as before.

All imports go through the public ``lib_python_comfy`` namespace.
"""
from __future__ import annotations

from lib_python_comfy import GraphBuilder, to_ui, txt2img, txt2audio


# ---------------------------------------------------------------------------
# txt2img scaffold — links and slot types
# ---------------------------------------------------------------------------


def test_txt2img_ui_links_have_no_unknown_types():
    """Every link's 6th element (type) is a real type, never 'UNKNOWN'."""
    g = txt2img(model="v1-5.safetensors", positive="cat", negative="ugly")
    result = to_ui(g)
    assert len(result["links"]) > 0
    for lnk in result["links"]:
        assert lnk[5] != "UNKNOWN"


def test_txt2img_ui_input_slot_types_are_real():
    """Every node's input slots carry a real (non-'UNKNOWN') type."""
    g = txt2img(model="v1-5.safetensors", positive="cat", negative="ugly")
    result = to_ui(g)
    for node in result["nodes"]:
        for slot in node["inputs"]:
            assert slot["type"] != "UNKNOWN", (node["type"], slot)


def test_txt2img_ui_output_slot_types_and_names_are_real():
    """Every node's output slots carry a real type and a type-derived name (not OUTPUT_N)."""
    g = txt2img(model="v1-5.safetensors", positive="cat", negative="ugly")
    result = to_ui(g)
    for node in result["nodes"]:
        for slot in node["outputs"]:
            assert slot["type"] != "UNKNOWN", (node["type"], slot)
            assert not slot["name"].startswith("OUTPUT_"), (node["type"], slot)


def test_txt2img_ckpt_output_types():
    """CheckpointLoaderSimple's outputs are named/typed MODEL, CLIP (x2 consumers), VAE."""
    g = txt2img(model="v1-5.safetensors", positive="cat", negative="ugly")
    result = to_ui(g)
    ckpt = next(n for n in result["nodes"] if n["type"] == "CheckpointLoaderSimple")
    types = [slot["type"] for slot in ckpt["outputs"]]
    names = [slot["name"] for slot in ckpt["outputs"]]
    assert types == ["MODEL", "CLIP", "VAE"]
    assert names == ["MODEL", "CLIP", "VAE"]


def test_txt2img_ksampler_input_types():
    """KSampler's input slots are typed MODEL/CONDITIONING/CONDITIONING/LATENT."""
    g = txt2img(model="v1-5.safetensors", positive="cat", negative="ugly")
    result = to_ui(g)
    ksampler = next(n for n in result["nodes"] if n["type"] == "KSampler")
    by_name = {slot["name"]: slot["type"] for slot in ksampler["inputs"]}
    assert by_name["model"] == "MODEL"
    assert by_name["positive"] == "CONDITIONING"
    assert by_name["negative"] == "CONDITIONING"
    assert by_name["latent_image"] == "LATENT"


# ---------------------------------------------------------------------------
# txt2img scaffold — widgets_values
# ---------------------------------------------------------------------------


def test_txt2img_checkpoint_widgets_values():
    """CheckpointLoaderSimple widgets_values == [ckpt_name]."""
    g = txt2img(model="my-model.safetensors", positive="cat", negative="ugly")
    result = to_ui(g)
    ckpt = next(n for n in result["nodes"] if n["type"] == "CheckpointLoaderSimple")
    assert ckpt["widgets_values"] == ["my-model.safetensors"]


def test_txt2img_clip_text_encode_widgets_values():
    """Each CLIPTextEncode node's widgets_values == [text] matching its literal prompt."""
    g = txt2img(model="m.safetensors", positive="a cat", negative="blurry")
    result = to_ui(g)
    clip_nodes = [n for n in result["nodes"] if n["type"] == "CLIPTextEncode"]
    texts = {tuple(n["widgets_values"]) for n in clip_nodes}
    assert texts == {("a cat",), ("blurry",)}


def test_txt2img_empty_latent_image_widgets_values():
    """EmptyLatentImage widgets_values == [width, height, batch_size]."""
    g = txt2img(model="m.safetensors", positive="cat", negative="", width=768, height=512)
    result = to_ui(g)
    latent = next(n for n in result["nodes"] if n["type"] == "EmptyLatentImage")
    assert latent["widgets_values"] == [768, 512, 1]


def test_txt2img_ksampler_widgets_values_with_fixed_placeholder():
    """KSampler widgets_values places the literal 'fixed' right after seed."""
    g = txt2img(
        model="m.safetensors",
        positive="cat",
        negative="",
        steps=30,
        cfg=8.5,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        seed=42,
    )
    result = to_ui(g)
    ksampler = next(n for n in result["nodes"] if n["type"] == "KSampler")
    assert ksampler["widgets_values"] == [42, "fixed", 30, 8.5, "dpmpp_2m", "karras", 1.0]


def test_txt2img_save_image_widgets_values():
    """SaveImage widgets_values == [filename_prefix]."""
    g = txt2img(model="m.safetensors", positive="cat", negative="")
    result = to_ui(g)
    save_image = next(n for n in result["nodes"] if n["type"] == "SaveImage")
    assert save_image["widgets_values"] == ["ComfyUI"]


def test_txt2img_node_size_is_a_list_not_a_dict():
    """node['size'] must be a [w, h] list (litegraph indexes size[0]/size[1]);
    a dict silently breaks link-curve anchor math (ticket #37)."""
    g = txt2img(model="m.safetensors", positive="cat", negative="")
    result = to_ui(g)
    assert len(result["nodes"]) > 0
    for node in result["nodes"]:
        assert isinstance(node["size"], list), (node["type"], node["size"])
        assert len(node["size"]) == 2
        assert node["size"] == [140, 80]


def test_txt2img_vae_decode_has_no_widgets_values_key():
    """VAEDecode has no widgets, so widgets_values must be omitted entirely."""
    g = txt2img(model="m.safetensors", positive="cat", negative="")
    result = to_ui(g)
    vae_decode = next(n for n in result["nodes"] if n["type"] == "VAEDecode")
    assert "widgets_values" not in vae_decode


# ---------------------------------------------------------------------------
# top-level last_node_id / last_link_id
# ---------------------------------------------------------------------------


def test_txt2img_last_node_id_and_last_link_id():
    """last_node_id/last_link_id match the actual max ids in the built graph."""
    g = txt2img(model="m.safetensors", positive="cat", negative="")
    result = to_ui(g)
    expected_last_node_id = max(n["id"] for n in result["nodes"])
    expected_last_link_id = max(lnk[0] for lnk in result["links"])
    assert result["last_node_id"] == expected_last_node_id
    assert result["last_link_id"] == expected_last_link_id
    # Sanity-check against the documented "7 nodes / 9 links" scaffold shape.
    assert expected_last_node_id == 7
    assert expected_last_link_id == 9


def test_empty_graph_last_ids_default_to_zero():
    """An empty graph's last_node_id/last_link_id default to 0 (no ids exist)."""
    g = GraphBuilder()
    result = to_ui(g)
    assert result["last_node_id"] == 0
    assert result["last_link_id"] == 0


# ---------------------------------------------------------------------------
# Unknown / custom node type — fallback preserved
# ---------------------------------------------------------------------------


def test_unknown_node_type_falls_back_to_unknown_sentinels():
    """A custom node type not in the registry keeps the legacy fallback behaviour."""
    g = GraphBuilder()
    src = g.add_node("SomeCustomNode", {"foo": "bar"})
    dst = g.add_node("AnotherCustomNode", {})
    g.link(src, 0, dst, "input_a")
    result = to_ui(g)

    src_node = next(n for n in result["nodes"] if n["id"] == src.id)
    dst_node = next(n for n in result["nodes"] if n["id"] == dst.id)

    # Output slot on the unknown-type source node.
    assert src_node["outputs"][0]["name"] == "OUTPUT_0"
    assert src_node["outputs"][0]["type"] == "UNKNOWN"
    assert "widgets_values" not in src_node

    # Input slot on the unknown-type destination node.
    assert dst_node["inputs"][0]["type"] == "UNKNOWN"
    assert "widgets_values" not in dst_node

    # Link type field.
    assert result["links"][0][5] == "UNKNOWN"


def test_known_class_type_unknown_input_name_falls_back():
    """A recognised class_type with an input name outside its registry falls back to UNKNOWN.

    (e.g. an extra hand-wired link into a slot name the registry doesn't know about)
    """
    g = GraphBuilder()
    src = g.add_node("Something")
    dst = g.add_node("KSampler", {})
    g.link(src, 0, dst, "not_a_real_ksampler_input")
    result = to_ui(g)
    dst_node = next(n for n in result["nodes"] if n["id"] == dst.id)
    assert dst_node["inputs"][0]["type"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# txt2audio scaffold smoke test
# ---------------------------------------------------------------------------


def test_txt2audio_ui_real_types_and_widgets():
    """txt2audio scaffold: EmptyLatentAudio/VAEDecodeAudio/SaveAudio emit real types and widgets."""
    g = txt2audio(model="stable_audio.safetensors", positive="ambient", seconds=30.0, batch_size=2)
    result = to_ui(g)

    latent = next(n for n in result["nodes"] if n["type"] == "EmptyLatentAudio")
    assert latent["outputs"][0]["type"] == "LATENT"
    assert latent["widgets_values"] == [30.0, 2]

    vae_decode_audio = next(n for n in result["nodes"] if n["type"] == "VAEDecodeAudio")
    by_name = {slot["name"]: slot["type"] for slot in vae_decode_audio["inputs"]}
    assert by_name["samples"] == "LATENT"
    assert by_name["vae"] == "VAE"
    assert vae_decode_audio["outputs"][0]["type"] == "AUDIO"
    assert "widgets_values" not in vae_decode_audio

    save_audio = next(n for n in result["nodes"] if n["type"] == "SaveAudio")
    audio_input = next(slot for slot in save_audio["inputs"] if slot["name"] == "audio")
    assert audio_input["type"] == "AUDIO"
    assert save_audio["widgets_values"] == ["ComfyUI"]

    # No UNKNOWN types anywhere in this scaffold's links.
    for lnk in result["links"]:
        assert lnk[5] != "UNKNOWN"
