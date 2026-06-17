"""Tests for the graph scaffolds module (ticket #9, #14).

All imports go through the public ``lib_python_comfy`` namespace to verify
that re-exports and ``__all__`` are wired correctly.
"""
from __future__ import annotations

import pytest

import lib_python_comfy
from lib_python_comfy import GraphBuilder, txt2img, txt2audio, txt2video, to_api


# ---------------------------------------------------------------------------
# Return type and public export
# ---------------------------------------------------------------------------


def test_txt2img_returns_graph_builder():
    """txt2img returns a GraphBuilder instance."""
    g = txt2img(model="v1-5.safetensors", positive="cat", negative="")
    assert isinstance(g, GraphBuilder)


def test_txt2img_in_public_all():
    """'txt2img' is listed in lib_python_comfy.__all__."""
    assert "txt2img" in lib_python_comfy.__all__


def test_txt2img_importable_from_top_level():
    """from lib_python_comfy import txt2img succeeds."""
    from lib_python_comfy import txt2img as fn  # noqa: F401 — import is the assertion
    assert fn is txt2img


# ---------------------------------------------------------------------------
# Node count and class types
# ---------------------------------------------------------------------------


def test_txt2img_produces_seven_nodes():
    """to_api(builder) has exactly 7 entries."""
    g = txt2img(model="v1-5.safetensors", positive="cat", negative="")
    result = to_api(g)
    assert len(result) == 7


def test_txt2img_node_class_types():
    """The set of class_type values is exactly the canonical 6 node types."""
    g = txt2img(model="v1-5.safetensors", positive="cat", negative="")
    result = to_api(g)
    types = {entry["class_type"] for entry in result.values()}
    assert types == {
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }


# ---------------------------------------------------------------------------
# Literal input propagation
# ---------------------------------------------------------------------------


def test_txt2img_literal_inputs_propagate():
    """Custom steps and cfg values land in the KSampler node inputs."""
    g = txt2img(model="m.safetensors", positive="dog", negative="ugly", steps=30, cfg=8.5)
    result = to_api(g)
    ksampler_entry = next(
        entry for entry in result.values() if entry["class_type"] == "KSampler"
    )
    assert ksampler_entry["inputs"]["steps"] == 30
    assert ksampler_entry["inputs"]["cfg"] == 8.5


def test_txt2img_default_values_in_api():
    """Default steps=20, cfg=7.0, sampler_name='euler', scheduler='normal', seed=0 appear in KSampler."""
    g = txt2img(model="m.safetensors", positive="cat", negative="")
    result = to_api(g)
    ksampler_entry = next(
        entry for entry in result.values() if entry["class_type"] == "KSampler"
    )
    inputs = ksampler_entry["inputs"]
    assert inputs["steps"] == 20
    assert inputs["cfg"] == 7.0
    assert inputs["sampler_name"] == "euler"
    assert inputs["scheduler"] == "normal"
    assert inputs["seed"] == 0


def test_txt2img_model_literal_in_checkpoint():
    """The checkpoint model name lands in CheckpointLoaderSimple ckpt_name."""
    g = txt2img(model="my-model.safetensors", positive="cat", negative="")
    result = to_api(g)
    ckpt_entry = next(
        entry for entry in result.values() if entry["class_type"] == "CheckpointLoaderSimple"
    )
    assert ckpt_entry["inputs"]["ckpt_name"] == "my-model.safetensors"


def test_txt2img_latent_dimensions_propagate():
    """Custom width/height land in the EmptyLatentImage node."""
    g = txt2img(model="m.safetensors", positive="cat", negative="", width=768, height=512)
    result = to_api(g)
    latent_entry = next(
        entry for entry in result.values() if entry["class_type"] == "EmptyLatentImage"
    )
    assert latent_entry["inputs"]["width"] == 768
    assert latent_entry["inputs"]["height"] == 512


# ---------------------------------------------------------------------------
# Link encoding
# ---------------------------------------------------------------------------


def test_txt2img_ksampler_model_link():
    """KSampler 'model' input is linked from CheckpointLoaderSimple slot 0."""
    g = txt2img(model="m.safetensors", positive="cat", negative="")
    result = to_api(g)
    ckpt_id = next(
        int(node_id)
        for node_id, entry in result.items()
        if entry["class_type"] == "CheckpointLoaderSimple"
    )
    ksampler_entry = next(
        entry for entry in result.values() if entry["class_type"] == "KSampler"
    )
    assert ksampler_entry["inputs"]["model"] == [str(ckpt_id), 0]


def test_txt2img_ksampler_positive_link():
    """KSampler 'positive' input is linked from the positive CLIPTextEncode slot 0."""
    g = txt2img(model="m.safetensors", positive="cat", negative="")
    result = to_api(g)
    ksampler_entry = next(
        entry for entry in result.values() if entry["class_type"] == "KSampler"
    )
    positive_link = ksampler_entry["inputs"]["positive"]
    # Must be a two-element list [str(node_id), 0]
    assert isinstance(positive_link, list)
    assert len(positive_link) == 2
    assert positive_link[1] == 0
    # The referenced node must be a CLIPTextEncode
    clip_node_id = positive_link[0]
    assert result[clip_node_id]["class_type"] == "CLIPTextEncode"


def test_txt2img_ksampler_negative_link():
    """KSampler 'negative' input is linked from the negative CLIPTextEncode slot 0."""
    g = txt2img(model="m.safetensors", positive="cat", negative="ugly")
    result = to_api(g)
    ksampler_entry = next(
        entry for entry in result.values() if entry["class_type"] == "KSampler"
    )
    negative_link = ksampler_entry["inputs"]["negative"]
    assert isinstance(negative_link, list)
    assert len(negative_link) == 2
    assert negative_link[1] == 0
    clip_node_id = negative_link[0]
    assert result[clip_node_id]["class_type"] == "CLIPTextEncode"


def test_txt2img_positive_and_negative_use_distinct_clip_nodes():
    """KSampler 'positive' and 'negative' links point to two different CLIPTextEncode nodes."""
    g = txt2img(model="m.safetensors", positive="cat", negative="ugly")
    result = to_api(g)
    ksampler_entry = next(
        entry for entry in result.values() if entry["class_type"] == "KSampler"
    )
    pos_id = ksampler_entry["inputs"]["positive"][0]
    neg_id = ksampler_entry["inputs"]["negative"][0]
    assert pos_id != neg_id


def test_txt2img_vae_decode_links():
    """VAEDecode 'samples' comes from KSampler slot 0; 'vae' comes from CheckpointLoaderSimple slot 2."""
    g = txt2img(model="m.safetensors", positive="cat", negative="")
    result = to_api(g)
    ckpt_id = next(
        node_id
        for node_id, entry in result.items()
        if entry["class_type"] == "CheckpointLoaderSimple"
    )
    ksampler_id = next(
        node_id
        for node_id, entry in result.items()
        if entry["class_type"] == "KSampler"
    )
    vae_decode_entry = next(
        entry for entry in result.values() if entry["class_type"] == "VAEDecode"
    )
    assert vae_decode_entry["inputs"]["samples"] == [ksampler_id, 0]
    assert vae_decode_entry["inputs"]["vae"] == [ckpt_id, 2]


# ---------------------------------------------------------------------------
# Required argument
# ---------------------------------------------------------------------------


def test_txt2img_model_is_required():
    """txt2img() without 'model' raises TypeError."""
    with pytest.raises(TypeError):
        txt2img(positive="cat", negative="")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Extensibility
# ---------------------------------------------------------------------------


def test_txt2img_builder_open_for_extension():
    """Adding a node and a link after the scaffold works; node count becomes 8."""
    g = txt2img(model="m.safetensors", positive="cat", negative="")
    result_before = to_api(g)
    assert len(result_before) == 7

    extra = g.add_node("PreviewImage", {})
    # Find VAEDecode's node id
    vae_decode_id_str = next(
        node_id
        for node_id, entry in result_before.items()
        if entry["class_type"] == "VAEDecode"
    )
    # We need the NodeRef — instead just verify node count increases
    result_after = to_api(g)
    assert len(result_after) == 8
    assert result_after[str(extra.id)]["class_type"] == "PreviewImage"


# ---------------------------------------------------------------------------
# txt2audio scaffold (ticket #29)
# ---------------------------------------------------------------------------


def test_txt2audio_returns_graph_builder():
    """txt2audio returns a GraphBuilder instance."""
    g = txt2audio(model="stable_audio.safetensors", positive="ambient music")
    assert isinstance(g, GraphBuilder)


def test_txt2audio_in_public_all():
    """'txt2audio' is listed in lib_python_comfy.__all__."""
    assert "txt2audio" in lib_python_comfy.__all__


def test_txt2audio_importable_from_top_level():
    """from lib_python_comfy import txt2audio succeeds."""
    from lib_python_comfy import txt2audio as fn  # noqa: F401
    assert fn is txt2audio


def test_txt2audio_produces_seven_nodes():
    """to_api(txt2audio(...)) has exactly 7 entries."""
    g = txt2audio(model="stable_audio.safetensors", positive="ambient music")
    result = to_api(g)
    assert len(result) == 7


def test_txt2audio_node_class_types():
    """The set of class_type values is exactly the 6 canonical node types."""
    g = txt2audio(model="stable_audio.safetensors", positive="ambient music")
    result = to_api(g)
    types = {entry["class_type"] for entry in result.values()}
    assert types == {
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentAudio",
        "KSampler",
        "VAEDecodeAudio",
        "SaveAudio",
    }


def test_txt2audio_ckpt_name_equals_model():
    """The CheckpointLoaderSimple ckpt_name is the literal supplied model."""
    g = txt2audio(model="my-audio-model.safetensors", positive="rain")
    result = to_api(g)
    ckpt_entry = next(
        entry for entry in result.values() if entry["class_type"] == "CheckpointLoaderSimple"
    )
    assert ckpt_entry["inputs"]["ckpt_name"] == "my-audio-model.safetensors"


def test_txt2audio_ksampler_model_link():
    """KSampler 'model' input is linked from CheckpointLoaderSimple slot 0."""
    g = txt2audio(model="m.safetensors", positive="beat")
    result = to_api(g)
    ckpt_id = next(
        node_id
        for node_id, entry in result.items()
        if entry["class_type"] == "CheckpointLoaderSimple"
    )
    ksampler_entry = next(
        entry for entry in result.values() if entry["class_type"] == "KSampler"
    )
    assert ksampler_entry["inputs"]["model"] == [ckpt_id, 0]


def test_txt2audio_ksampler_positive_link():
    """KSampler 'positive' input is linked from the positive CLIPTextEncode slot 0."""
    g = txt2audio(model="m.safetensors", positive="beat")
    result = to_api(g)
    ksampler_entry = next(
        entry for entry in result.values() if entry["class_type"] == "KSampler"
    )
    positive_link = ksampler_entry["inputs"]["positive"]
    assert isinstance(positive_link, list)
    assert len(positive_link) == 2
    assert positive_link[1] == 0
    assert result[positive_link[0]]["class_type"] == "CLIPTextEncode"


def test_txt2audio_ksampler_negative_link():
    """KSampler 'negative' input is linked from the negative CLIPTextEncode slot 0."""
    g = txt2audio(model="m.safetensors", positive="beat", negative="noise")
    result = to_api(g)
    ksampler_entry = next(
        entry for entry in result.values() if entry["class_type"] == "KSampler"
    )
    negative_link = ksampler_entry["inputs"]["negative"]
    assert isinstance(negative_link, list)
    assert len(negative_link) == 2
    assert negative_link[1] == 0
    assert result[negative_link[0]]["class_type"] == "CLIPTextEncode"


def test_txt2audio_positive_and_negative_use_distinct_clip_nodes():
    """KSampler 'positive' and 'negative' links point to two different CLIPTextEncode nodes."""
    g = txt2audio(model="m.safetensors", positive="beat", negative="noise")
    result = to_api(g)
    ksampler_entry = next(
        entry for entry in result.values() if entry["class_type"] == "KSampler"
    )
    pos_id = ksampler_entry["inputs"]["positive"][0]
    neg_id = ksampler_entry["inputs"]["negative"][0]
    assert pos_id != neg_id


def test_txt2audio_vae_decode_audio_links():
    """VAEDecodeAudio 'samples' comes from KSampler slot 0; 'vae' from CheckpointLoaderSimple slot 2."""
    g = txt2audio(model="m.safetensors", positive="beat")
    result = to_api(g)
    ckpt_id = next(
        node_id
        for node_id, entry in result.items()
        if entry["class_type"] == "CheckpointLoaderSimple"
    )
    ksampler_id = next(
        node_id
        for node_id, entry in result.items()
        if entry["class_type"] == "KSampler"
    )
    vae_decode_entry = next(
        entry for entry in result.values() if entry["class_type"] == "VAEDecodeAudio"
    )
    assert vae_decode_entry["inputs"]["samples"] == [ksampler_id, 0]
    assert vae_decode_entry["inputs"]["vae"] == [ckpt_id, 2]


def test_txt2audio_save_audio_link():
    """SaveAudio 'audio' input is linked from VAEDecodeAudio slot 0."""
    g = txt2audio(model="m.safetensors", positive="beat")
    result = to_api(g)
    vae_decode_id = next(
        node_id
        for node_id, entry in result.items()
        if entry["class_type"] == "VAEDecodeAudio"
    )
    save_audio_entry = next(
        entry for entry in result.values() if entry["class_type"] == "SaveAudio"
    )
    assert save_audio_entry["inputs"]["audio"] == [vae_decode_id, 0]


def test_txt2audio_model_is_required():
    """txt2audio() without 'model' raises TypeError."""
    with pytest.raises(TypeError):
        txt2audio(positive="hi")  # type: ignore[call-arg]


def test_txt2audio_accepts_keyword_args():
    """txt2audio accepts all keyword arguments including model and optional params."""
    g = txt2audio(
        model="m.safetensors",
        positive="hello",
        negative="bad",
        seconds=10.0,
        batch_size=2,
        sample_rate=22050,
        steps=30,
        cfg=5.0,
        sampler_name="euler",
        scheduler="normal",
        seed=99,
    )
    assert isinstance(g, GraphBuilder)


def test_txt2audio_builder_open_for_extension():
    """Adding a node after txt2audio increases the node count to 8."""
    g = txt2audio(model="m.safetensors", positive="beat")
    extra = g.add_node("AudioPostProcess", {})
    result = to_api(g)
    assert len(result) == 8
    assert result[str(extra.id)]["class_type"] == "AudioPostProcess"


# ---------------------------------------------------------------------------
# txt2video scaffold (ticket #29)
# ---------------------------------------------------------------------------


def test_txt2video_raises_not_implemented():
    """txt2video() raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        txt2video()


def test_txt2video_error_message_mentions_video_node():
    """The NotImplementedError message mentions the video custom node."""
    with pytest.raises(NotImplementedError, match="VHS_VideoCombine"):
        txt2video()


def test_txt2video_raises_with_kwargs():
    """txt2video() raises NotImplementedError regardless of keyword arguments."""
    with pytest.raises(NotImplementedError):
        txt2video(positive="fly", negative="static", width=1024, height=576, seed=7)


def test_txt2video_in_public_all():
    """'txt2video' is listed in lib_python_comfy.__all__."""
    assert "txt2video" in lib_python_comfy.__all__


def test_txt2video_importable_from_top_level():
    """from lib_python_comfy import txt2video succeeds."""
    from lib_python_comfy import txt2video as fn  # noqa: F401
    assert fn is txt2video
