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


def txt2audio(
    *,
    model: str,
    positive: str,
    negative: str = "",
    seconds: float = 47.0,
    batch_size: int = 1,
    sample_rate: int = 44100,
    steps: int = 20,
    cfg: float = 3.5,
    sampler_name: str = "dpmpp_3m_sde_gpu",
    scheduler: str = "exponential",
    seed: int = 0,
    clip_name: str | None = None,
    clip_type: str = "stable_audio",
    seconds_start: float = 0.0,
) -> GraphBuilder:
    """Return a :class:`GraphBuilder` pre-wired for a Stable Audio pipeline.

    By default, builds the canonical ComfyUI Stable Audio graph, routing the
    text encoder from the checkpoint itself::

        CheckpointLoaderSimple
            ├─[MODEL]──────────────► KSampler ─[LATENT]─► VAEDecodeAudio ─[AUDIO]─► SaveAudio
            ├─[CLIP]───► CLIPTextEncode(+) ─[CONDITIONING]─► KSampler
            ├─[CLIP]───► CLIPTextEncode(-) ─[CONDITIONING]─► KSampler
            └─[VAE]────────────────────────────────────────► VAEDecodeAudio
        EmptyLatentAudio ─[LATENT]─► KSampler

    When *clip_name* is supplied, a ``CLIPLoader`` node is inserted, both
    ``CLIPTextEncode`` nodes are routed from it instead of the checkpoint,
    and a ``ConditioningStableAudio`` stage is added between them and
    ``KSampler``::

        CLIPLoader
            └─[CLIP]───► CLIPTextEncode(+) ─┐
        CLIPLoader                          ├─[CONDITIONING]─► ConditioningStableAudio ─┐
            └─[CLIP]───► CLIPTextEncode(-) ─┘                                           │
        CheckpointLoaderSimple                                                          │
            ├─[MODEL]──────────────────────────────────────► KSampler ◄─────────────────┘
            └─[VAE]────────────────────────────────────────► VAEDecodeAudio ─[AUDIO]─► SaveAudio
        EmptyLatentAudio ─[LATENT]─► KSampler

    The returned builder is open for further extension — callers may append
    additional nodes and links before serialising with :func:`to_api` or
    :func:`to_ui`.

    .. note::

        Checkpoints that do not bundle a text encoder (e.g.
        ``stable_audio_open_1.0.safetensors``) leave ``CheckpointLoaderSimple``
        slot 1 empty. Because the default graph routes both
        ``CLIPTextEncode.clip`` inputs from that slot, such a checkpoint fails
        at execution time with ComfyUI's ``clip input is invalid: None``. Pass
        ``clip_name`` (and ``clip_type`` if the encoder is not a Stable Audio
        T5) to load the text encoder from a separate ``CLIPLoader``; this also
        inserts the ``ConditioningStableAudio`` stage. The library cannot
        detect this from the checkpoint filename and deliberately emits no
        warning.

    Args:
        model: Checkpoint filename as recognised by ComfyUI
               (e.g. ``"stable_audio_open_1.0.safetensors"``).
        positive: Positive prompt text fed into CLIPTextEncode.
        negative: Negative prompt text fed into CLIPTextEncode (default ``""``).
        seconds: Duration of the generated audio in seconds (default 47.0).
        batch_size: Number of audio samples to generate (default 1).
        sample_rate: Sample rate in Hz (default 44100).
        steps: Number of KSampler denoising steps (default 20).
        cfg: Classifier-free guidance scale (default 3.5).
        sampler_name: KSampler algorithm name (default ``"dpmpp_3m_sde_gpu"``).
        scheduler: KSampler scheduler name (default ``"exponential"``).
        seed: RNG seed for the KSampler (default 0).
        clip_name: Text encoder filename to load via a separate
            ``CLIPLoader`` node (e.g. ``"t5-base.safetensors"``). When
            omitted (``None``, the default) or an empty string, both
            ``CLIPTextEncode`` nodes are routed from the checkpoint's own
            CLIP output instead — an empty string is equivalent to omitting
            it. A whitespace-only string is treated as supplied and builds a
            ``CLIPLoader`` with that (invalid) filename, which ComfyUI will
            reject at execution time.
        clip_type: ``CLIPLoader`` encoder type widget, only used when
            *clip_name* is supplied (default ``"stable_audio"``).
        seconds_start: Start offset in seconds fed to the
            ``ConditioningStableAudio`` stage, only used when *clip_name* is
            supplied (default 0.0).

    Returns:
        A configured :class:`GraphBuilder`: 7 nodes / 9 links by default;
        9 nodes / 11 links when ``clip_name`` is supplied.
    """
    g = GraphBuilder()

    if clip_name:
        clip_loader = g.add_node("CLIPLoader", {"clip_name": clip_name, "type": clip_type})
        ckpt = g.add_node("CheckpointLoaderSimple", {"ckpt_name": model})
        clip_pos = g.add_node("CLIPTextEncode", {"text": positive})
        clip_neg = g.add_node("CLIPTextEncode", {"text": negative})
        cond = g.add_node(
            "ConditioningStableAudio",
            {"seconds_start": seconds_start, "seconds_total": seconds},
        )
        latent = g.add_node(
            "EmptyLatentAudio",
            {"seconds": seconds, "batch_size": batch_size, "sample_rate": sample_rate},
        )
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
        vae_decode = g.add_node("VAEDecodeAudio", {})
        save_audio = g.add_node("SaveAudio", {"filename_prefix": "ComfyUI"})

        # CheckpointLoaderSimple outputs: 0=MODEL, 1=CLIP, 2=VAE
        g.link(ckpt, 0, ksampler, "model")

        # CLIPLoader outputs: 0=CLIP
        g.link(clip_loader, 0, clip_pos, "clip")
        g.link(clip_loader, 0, clip_neg, "clip")

        # CLIPTextEncode outputs: 0=CONDITIONING
        g.link(clip_pos, 0, cond, "positive")
        g.link(clip_neg, 0, cond, "negative")

        # ConditioningStableAudio outputs: 0=CONDITIONING (positive), 1=CONDITIONING (negative)
        g.link(cond, 0, ksampler, "positive")
        g.link(cond, 1, ksampler, "negative")

        # EmptyLatentAudio outputs: 0=LATENT
        g.link(latent, 0, ksampler, "latent_image")

        # KSampler outputs: 0=LATENT
        g.link(ksampler, 0, vae_decode, "samples")

        # CheckpointLoaderSimple slot 2 = VAE
        g.link(ckpt, 2, vae_decode, "vae")

        # VAEDecodeAudio outputs: 0=AUDIO
        g.link(vae_decode, 0, save_audio, "audio")

        return g

    ckpt = g.add_node("CheckpointLoaderSimple", {"ckpt_name": model})
    clip_pos = g.add_node("CLIPTextEncode", {"text": positive})
    clip_neg = g.add_node("CLIPTextEncode", {"text": negative})
    latent = g.add_node(
        "EmptyLatentAudio",
        {"seconds": seconds, "batch_size": batch_size, "sample_rate": sample_rate},
    )
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
    vae_decode = g.add_node("VAEDecodeAudio", {})
    save_audio = g.add_node("SaveAudio", {"filename_prefix": "ComfyUI"})

    # CheckpointLoaderSimple outputs: 0=MODEL, 1=CLIP, 2=VAE
    g.link(ckpt, 0, ksampler, "model")
    g.link(ckpt, 1, clip_pos, "clip")
    g.link(ckpt, 1, clip_neg, "clip")

    # CLIPTextEncode outputs: 0=CONDITIONING
    g.link(clip_pos, 0, ksampler, "positive")
    g.link(clip_neg, 0, ksampler, "negative")

    # EmptyLatentAudio outputs: 0=LATENT
    g.link(latent, 0, ksampler, "latent_image")

    # KSampler outputs: 0=LATENT
    g.link(ksampler, 0, vae_decode, "samples")

    # CheckpointLoaderSimple slot 2 = VAE
    g.link(ckpt, 2, vae_decode, "vae")

    # VAEDecodeAudio outputs: 0=AUDIO
    g.link(vae_decode, 0, save_audio, "audio")

    return g


def txt2video(
    *,
    positive: str = "",
    negative: str = "",
    width: int = 512,
    height: int = 512,
    seed: int = 0,
) -> GraphBuilder:
    """Return a :class:`GraphBuilder` pre-wired for a text-to-video pipeline.

    .. note::

        This function always raises :exc:`NotImplementedError` because
        text-to-video generation requires a custom node
        (e.g. ``VHS_VideoCombine`` from ComfyUI-VideoHelper-Suite) that is
        **not** present in a stock ComfyUI install.

    To build a video graph, install a compatible video node and wire it
    directly with :class:`~lib_python_comfy.graph.GraphBuilder`.

    Args:
        positive: Positive prompt text.
        negative: Negative prompt text.
        width: Output width in pixels (default 512).
        height: Output height in pixels (default 512).
        seed: RNG seed (default 0).

    Raises:
        NotImplementedError: Always — a compatible video node must be
            installed and wired manually.
    """
    raise NotImplementedError(
        "txt2video requires a custom video-generation node (e.g. "
        "ComfyUI-VideoHelper-Suite / VHS_VideoCombine) that is not present "
        "in a stock ComfyUI install. "
        "Build the graph directly with GraphBuilder after installing a "
        "compatible video node."
    )
