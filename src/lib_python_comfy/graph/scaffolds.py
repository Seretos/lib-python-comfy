"""Graph scaffolds — convenience wrappers that pre-wire common ComfyUI pipelines.

Each scaffold builds a fully-connected :class:`~lib_python_comfy.graph.GraphBuilder`
for a standard workflow and returns it so the caller can still extend or
inspect it.  All node construction goes through :meth:`GraphBuilder.add_node`
and :meth:`GraphBuilder.link`; no internal structures are touched directly.
"""
from __future__ import annotations

from .builder import GraphBuilder


def txt2img(
    *,
    model: str,
    positive: str,
    negative: str,
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg: float = 7.0,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    seed: int = 0,
) -> GraphBuilder:
    """Return a :class:`GraphBuilder` pre-wired for a text-to-image pipeline.

    Builds the canonical ComfyUI txt2img graph::

        CheckpointLoaderSimple
            ├─[MODEL]──────────────► KSampler ─[LATENT]─► VAEDecode ─[IMAGE]─► SaveImage
            ├─[CLIP]───► CLIPTextEncode(+) ─[CONDITIONING]─► KSampler
            ├─[CLIP]───► CLIPTextEncode(-) ─[CONDITIONING]─► KSampler
            └─[VAE]────────────────────────────────────────► VAEDecode
        EmptyLatentImage ─[LATENT]─► KSampler

    The returned builder is open for further extension — callers may append
    additional nodes and links before serialising with :func:`to_api` or
    :func:`to_ui`.

    Args:
        model: Checkpoint filename as recognised by ComfyUI
               (e.g. ``"v1-5-pruned-emaonly.safetensors"``).
        positive: Positive prompt text fed into CLIPTextEncode.
        negative: Negative prompt text fed into CLIPTextEncode.
        width: Latent width in pixels (default 512).
        height: Latent height in pixels (default 512).
        steps: Number of KSampler denoising steps (default 20).
        cfg: Classifier-free guidance scale (default 7.0).
        sampler_name: KSampler algorithm name (default ``"euler"``).
        scheduler: KSampler scheduler name (default ``"normal"``).
        seed: RNG seed for the KSampler (default 0).

    Returns:
        A configured :class:`GraphBuilder` with 7 nodes and 9 links.
    """
    g = GraphBuilder()

    ckpt = g.add_node("CheckpointLoaderSimple", {"ckpt_name": model})
    clip_pos = g.add_node("CLIPTextEncode", {"text": positive})
    clip_neg = g.add_node("CLIPTextEncode", {"text": negative})
    latent = g.add_node("EmptyLatentImage", {"width": width, "height": height, "batch_size": 1})
    ksampler = g.add_node(
        "KSampler",
        {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    )
    vae_decode = g.add_node("VAEDecode", {})
    save_image = g.add_node("SaveImage", {"filename_prefix": "ComfyUI"})

    # CheckpointLoaderSimple outputs: 0=MODEL, 1=CLIP, 2=VAE
    g.link(ckpt, 0, ksampler, "model")
    g.link(ckpt, 1, clip_pos, "clip")
    g.link(ckpt, 1, clip_neg, "clip")

    # CLIPTextEncode outputs: 0=CONDITIONING
    g.link(clip_pos, 0, ksampler, "positive")
    g.link(clip_neg, 0, ksampler, "negative")

    # EmptyLatentImage outputs: 0=LATENT
    g.link(latent, 0, ksampler, "latent_image")

    # KSampler outputs: 0=LATENT
    g.link(ksampler, 0, vae_decode, "samples")

    # CheckpointLoaderSimple slot 2 = VAE
    g.link(ckpt, 2, vae_decode, "vae")

    # VAEDecode outputs: 0=IMAGE
    g.link(vae_decode, 0, save_image, "images")

    return g
