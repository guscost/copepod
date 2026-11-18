"""SANA-Streaming video-to-video backend.

Streams webcam frames through the SANA-Streaming V2V model (default prompt:
replace logos/ads with black squares) and returns the processed frames.

Architecture notes:
  * The VAE keeps per-layer streaming caches in process memory, so only a
    single producer stream can run safely.
  * All GPU work must be serialized on a single thread (the server's GPU
    worker) so the model, the VAE caches, and the DiT KV cache are never
    touched concurrently.
  * The DiT sampler keeps its autoregressive KV cache alive across calls
    (``IncrementalStreamingSampler``), which is what makes the output
    temporally consistent instead of frame-by-frame flicker.
"""

import copy
import functools
import io
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# SANA-Streaming environment setup (must run before importing torch/diffusion).
# Native PyTorch SDPA (FlashAttention backend) is preferred over xFormers: on
# Blackwell (sm_120) the core SDPA flash kernels are the maintained path, while
# xFormers lags new GPU archs. flash-attn still takes precedence in the V2V
# softmax attention where it is wired in directly. Set DISABLE_XFORMERS=0 to
# re-enable xFormers (e.g. for the V2V cross-attention packing path).
os.environ.setdefault("DISABLE_XFORMERS", "1")
os.environ.setdefault("USE_CHUNKWISE_GDN", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Persistent kernel/engine caches so Triton compiles, Inductor artifacts, and
# TensorRT engine builds are paid once across server restarts.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/sana_inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.cache/sana_triton"))

BACKEND_ROOT = Path(__file__).resolve().parent
# The Sana repo is not tracked in this repository; it must exist locally.
# Override with the SANA_ROOT environment variable if it lives elsewhere.
SANA_ROOT = Path(os.environ.get("SANA_ROOT", BACKEND_ROOT / "Sana-main")).resolve()
if (SANA_ROOT / "diffusion").is_dir() and str(SANA_ROOT) not in sys.path:
    sys.path.insert(0, str(SANA_ROOT))

try:
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image, ImageOps

    import diffusion.model.nets  # noqa: F401  (registers model/attention classes)
    import pyrallis
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    from diffusion.model.builder import build_model, get_vae
    from diffusion.model.ltx2.streaming_decoder import CausalVaeStreamingDecoder
    from diffusion.model.utils import get_weight_dtype
    from diffusion.scheduler.sana_streaming_sampler import SANAStreamingSampler
    from diffusion.utils.config import (
        AEConfig,
        ModelVideoConfig,
        SchedulerConfig,
        TextEncoderConfig,
        model_v2v_init_config,
    )
    from tools.download import find_model
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Failed to import the SANA-Streaming stack: {e}\n\n"
        "Setup:\n"
        f"  1. The Sana repo must exist at {SANA_ROOT}\n"
        "     (clone it there, or point the SANA_ROOT env var at it).\n"
        "  2. Install Sana's dependencies (Python 3.11, CUDA toolkit):\n"
        "       cd backend/Sana-main && bash environment_setup.sh\n"
        "     or minimally:  pip install -e backend/Sana-main\n"
        "  3. Install the server deps:  pip install -r backend/requirements.txt"
    ) from e

# Force the native SDPA FlashAttention backend on CUDA (the default, but make
# it explicit; SDPA still falls back to mem-efficient/math per-op when flash
# can't handle a shape, e.g. arbitrary attention masks).
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

STREAMING_REPO_ID = "Efficient-Large-Model/SANA-Streaming"
DEFAULT_PROMPT = "Replace all corporate logos, and all still or video advertising, with black squares"
CONFIG_PATH = BACKEND_ROOT / "sana_streaming_2b_720p.yaml"

# Load knobs (env-overridable). Lower these if the GPU/PSU can't sustain the
# load — e.g. transient power spikes on an RTX 5090 can trip the whole machine:
#   FRAME_WIDTH=832 FRAME_HEIGHT=480   # 480p bucket, ~2x less DiT/VAE work
#   DENOISE_STEPS=2                    # halves DiT cost per chunk
#   FRAME_SKIP=2                       # process 1 of every 2 input frames
# Both frame dims must be divisible by the VAE spatial stride (32), which
# 1280x720 is not.
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "704"))
DENOISE_STEPS = int(os.environ.get("DENOISE_STEPS", "4"))  # long_streaming default
# Process only every Nth input frame; skipped frames are answered with the
# last output, lowering the sustained processing framerate.
FRAME_SKIP = max(1, int(os.environ.get("FRAME_SKIP", "1")))
JPEG_QUALITY = 80
# If the GPU falls behind by more than this many unprocessed input frames,
# drop the oldest ones (a jump cut) instead of growing latency without bound.
MAX_PENDING_FRAMES = 48

# Experimental Torch-TensorRT compilation of hot components. Off by default;
# enable with a comma-separated list, e.g. TENSORRT=vae,text_encoder (or
# TENSORRT=all). Requires a torch-tensorrt build matching the installed
# torch/CUDA. The DiT is deliberately not compilable this way: its stateful
# KV cache and Triton GDN kernels don't survive graph capture (an AOT
# dynamo compile with the KV cache as an explicit input would need the
# sampler restructured to be purely functional). A component whose compiled
# call ever fails stays on eager for the rest of the session.
TENSORRT_COMPONENTS = {c.strip() for c in os.environ.get("TENSORRT", "").split(",") if c.strip()}
if TENSORRT_COMPONENTS - {"vae", "text_encoder", "all"}:
    raise SystemExit(
        f"Unknown TENSORRT component(s): {sorted(TENSORRT_COMPONENTS - {'vae', 'text_encoder', 'all'})} "
        "(expected a comma-separated subset of: vae, text_encoder, all)"
    )

# TorchAO microscaling quantization of the DiT (opt-in, TORCHAO_QUANT=1).
# Bulk FFN-out/attention GEMMs go to Blackwell NVFP4 weight-only (block size
# 16); the sensitive FFN fc1 and GDN linear layers go to MXFP8 dynamic
# activation + weight (block size 32). Conditioning, embeddings, output heads,
# and norms stay in bf16. NVFP4/MXFP8 use torchao's prototype mx_formats API
# (needs a recent torchao and a Blackwell GPU, sm_100+/sm_120); older
# torchao/GPUs degrade to int4 weight-only + int8 dynamic instead.
TORCHAO_QUANT = os.environ.get("TORCHAO_QUANT", "0") == "1"

_trt_logged: set[str] = set()


def _trt_compile(fn, component: str, label: str):
    """Wrap ``fn`` in torch.compile(backend="torch_tensorrt") when its component is enabled.

    Compilation is lazy (it happens on the first call), so the wrapper catches
    a failure from any compiled call and permanently falls back to the eager
    function — a bad graph can never kill the stream.
    """
    if not (TENSORRT_COMPONENTS & {component, "all"}):
        return fn
    try:
        # The import registers "torch_tensorrt" as a torch.compile backend;
        # without it torch.compile raises InvalidBackend.
        import torch_tensorrt  # noqa: F401
    except ImportError:
        if component not in _trt_logged:
            _trt_logged.add(component)
            print(
                f"[tensorrt] {component}: torch-tensorrt not installed; staying eager "
                "(install a torch-tensorrt build matching your torch/CUDA to enable)",
                flush=True,
            )
        return fn
    # Mirror the recommended Blackwell (sm_120) dynamo settings: bf16 with fp8
    # (and fp4 where the torch build exposes it) allowed, explicit typing,
    # highest optimization/tactic-search level, CUDA graphs to cut WSL2/WDDM
    # launch overhead, and a persistent engine cache so recompiles across
    # restarts are skipped. fp8/fp4 execution for the DiT itself comes from
    # the TorchAO microscaling path (TORCHAO_QUANT=1), not TensorRT.
    enabled_precisions = {torch.bfloat16, torch.float8_e4m3fn}
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is not None:
        enabled_precisions.add(fp4_dtype)
    trt_options = {
        "enabled_precisions": enabled_precisions,
        "use_explicit_typing": True,
        "optimization_level": 4,
        "cudagraphs": True,
        "cache_built_engines": True,
        "reuse_cached_engines": True,
        "engine_cache_dir": os.path.expanduser("~/.cache/sana_trt_engines"),
    }
    compiled = torch.compile(fn, backend="torch_tensorrt", dynamic=False, options=trt_options)
    failed = False

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal failed
        if failed:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as e:
            failed = True
            print(
                f"[tensorrt] {label} failed under TensorRT ({e}); "
                "eager for the rest of the session",
                flush=True,
            )
            return fn(*args, **kwargs)

    if component not in _trt_logged:
        _trt_logged.add(component)
        print(f"[tensorrt] {component}: Torch-TensorRT enabled (compiles on first call)", flush=True)
    return wrapper


def _sana_blackwell_filter(module: torch.nn.Module, fqn: str) -> bool:
    """NVFP4 (block size 16) filter: bulk FFN-out/attention GEMMs only."""
    if not isinstance(module, torch.nn.Linear):
        return False
    # NVFP4 kernels require both weight dims divisible by the 16-wide block;
    # layers that fail this (e.g. small gate projections) stay in bf16.
    if module.in_features % 16 or module.out_features % 16:
        return False
    # Protect conditioning, embeddings, output heads, GDN state projections,
    # and the camera-control QKV projections (proj_cam) — Sana fuses those at
    # runtime with torch.cat([q.weight, k.weight, v.weight]), and NVFP4Tensor
    # doesn't implement aten.cat.
    protected_tokens = ("t_embedder", "y_embedder", "proj_out", "gdn", "delta_net", "norm", "proj_cam")
    if any(tok in fqn.lower() for tok in protected_tokens):
        return False
    # Keep FFN input projections in higher precision to prevent temporal flicker
    if "ffn.fc1" in fqn or "mlp.fc1" in fqn:
        return False
    return True


def _mxfp8_filter(module: torch.nn.Module, fqn: str) -> bool:
    """MXFP8 (block size 32) filter: sensitive FFN fc1 & GDN linear layers."""
    return (
        isinstance(module, torch.nn.Linear)
        and ("ffn.fc1" in fqn or "gdn" in fqn.lower())
        # MX blocks tile the reduction dim by 32; skip layers that don't fit.
        and module.in_features % 32 == 0
    )


def _quantize_dit_microscaling(model: torch.nn.Module) -> None:
    """Apply TorchAO microscaling quantization to the DiT in place.

    Prefers the true Blackwell microscaling path (NVFP4 weight-only +
    MXFP8 dynamic, from torchao's prototype mx_formats API) and degrades to
    int4/int8 (stable torchao API) when the prototype configs are unavailable.
    """
    try:
        from torchao import __version__ as ao_version
        from torchao.quantization import quantize_
    except Exception as e:  # not just ImportError: broken CUDA extensions raise OSError/RuntimeError
        print(
            f"[quant] torchao not importable ({e}); DiT stays in bf16 "
            "(install torchao matching your torch/CUDA to enable TORCHAO_QUANT=1)",
            flush=True,
        )
        return

    nvfp4_config = None
    mxfp8_config = None
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) >= (10, 0):
        try:
            # The microscaling configs live in the prototype mx_formats API
            # (they are NOT exported from torchao.quantization):
            #   NVFP4WeightOnlyConfig            -> NVFP4 weight-only, block size 16
            #   MXDynamicActivationMXWeightConfig-> MXFP8 dynamic act+weight, block size 32
            #     (defaults: activation_dtype = weight_dtype = float8_e4m3fn)
            from torchao.prototype.mx_formats.inference_workflow import (
                MXDynamicActivationMXWeightConfig,
                NVFP4WeightOnlyConfig,
            )

            nvfp4_config = NVFP4WeightOnlyConfig()
            mxfp8_config = MXDynamicActivationMXWeightConfig(block_size=32)
        except Exception as e:
            print(
                f"[quant] torchao {ao_version} lacks the prototype MX/NVFP4 configs ({e}); "
                "upgrade torchao for the Blackwell microscaling path — falling back to int4/int8",
                flush=True,
            )
    else:
        print(
            f"[quant] NVFP4/MXFP8 microscaling needs a Blackwell GPU (sm_100+); "
            f"this is sm_{major}{minor} — falling back to int4/int8",
            flush=True,
        )

    if nvfp4_config is None:
        from torchao.quantization import (
            Int4WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
        )

        nvfp4_config = Int4WeightOnlyConfig()
        mxfp8_config = Int8DynamicActivationInt8WeightConfig()

    try:
        # 1. Bulk FFN output/attention GEMMs -> NVFP4 weight-only (block size 16)
        quantize_(model, nvfp4_config, filter_fn=_sana_blackwell_filter)
        # 2. Sensitive FFN fc1 & GDN linear layers -> MXFP8 dynamic (block size 32)
        quantize_(model, mxfp8_config, filter_fn=_mxfp8_filter)
    except Exception as e:
        print(f"[quant] TorchAO quantization failed ({e}); continuing anyway", flush=True)
        return
    print(
        f"[quant] DiT quantized (torchao {ao_version}): "
        f"{type(nvfp4_config).__name__} (bulk GEMMs) + {type(mxfp8_config).__name__} (fc1/GDN)",
        flush=True,
    )


@dataclass
class StreamingInferenceConfig:
    model: ModelVideoConfig
    vae: AEConfig
    text_encoder: TextEncoderConfig
    scheduler: SchedulerConfig
    work_dir: str = "output"


def load_text_encoder(name: str, device, attn_implementation: str):
    """Gemma text encoder load with an explicit attention backend.

    Mirrors the gemma branch of Sana's ``get_tokenizer_and_text_encoder`` but
    passes ``attn_implementation`` through, so the attention backend can be
    chosen (``flash_attention_2`` by default; ``eager``/``sdpa``/``
    flash_attention_2`` via the TEXT_ENCODER_ATTN env var).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = name if "/" in name else f"Efficient-Large-Model/{name}"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation
        )
        .get_decoder()
        .to(device)
    )
    return tokenizer, text_encoder


class IncrementalStreamingSampler(SANAStreamingSampler):
    """SANAStreamingSampler variant whose KV cache survives between calls.

    The stock sampler denoises a complete video inside a single ``sample()``
    call, building and discarding the KV cache internally (and refusing inputs
    of ``base_chunk_frames`` latent frames or fewer). A live websocket stream
    never "ends", so this subclass denoises one chunk of latent frames per
    call while the autoregressive cache persists until ``reset_stream()``.
    """

    def __init__(self, *args, autocast_dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_info = self.model_kwargs.pop("data_info", {})
        self._autocast_dtype = autocast_dtype
        self.reset_stream()

    def reset_stream(self):
        self._kv_cache = []
        self._chunk_indices = [0]

    def _forward(self, *args, **kwargs):
        device = self.condition.device
        if self._autocast_dtype is not None and device.type == "cuda":
            # Mirror the reference inference script, which runs under an
            # accelerate mixed-precision autocast context.
            with torch.autocast(device.type, dtype=self._autocast_dtype):
                return self.model(*args, **kwargs)
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def sample_chunk(self, latents, image_vae_embeds, steps=DENOISE_STEPS):
        """Denoise one new chunk of latent frames.

        Args:
            latents: fresh noise for the new chunk, ``[B, C, F, H, W]``.
            image_vae_embeds: source-video latents for the same ``F`` frames.

        Returns:
            The denoised chunk latents, ``[B, C, F, H, W]``.
        """
        device = self.condition.device
        do_cfg = self.cfg_scale > 1
        _, _, chunk_frames, height, width = latents.shape
        self._spatial_hw = height * width

        chunk_idx = len(self._kv_cache)
        self._chunk_indices.append(self._chunk_indices[-1] + chunk_frames)
        self._kv_cache.append([[None] * 6 for _ in range(self.num_model_blocks)])

        chunk_kv_cache, _, sink_num, num_cached_frames = self.accumulate_kv_cache(self._kv_cache, chunk_idx)

        prompt_embeds = self.condition
        if prompt_embeds.dim() == 4:
            prompt_embeds = prompt_embeds[:, 0]
        if do_cfg:
            uncond = self.uncondition
            if uncond.dim() == 4:
                uncond = uncond[:, 0]
            prompt_embeds = torch.cat([uncond, prompt_embeds], dim=0)

        scheduler = FlowMatchEulerDiscreteScheduler(shift=self.flow_shift)
        timesteps = self._timesteps_for_steps(scheduler, steps, device)

        start_f = self._chunk_indices[chunk_idx]
        end_f = self._chunk_indices[chunk_idx + 1]
        cache_start_chunk_idx = max(chunk_idx - self.num_cached_blocks, 0) if self.num_cached_blocks > 0 else 0

        frame_index = None
        if sink_num > 0:
            sink_fi = torch.arange(sink_num, device=device)
            non_sink_count = num_cached_frames - sink_num + chunk_frames
            window_start_f = end_f - non_sink_count
            remaining_fi = torch.arange(window_start_f, end_f, device=device)
            frame_index = torch.cat([sink_fi, remaining_fi], dim=0)
            rope_start_f = 0
            rope_end_f = end_f
        else:
            rope_start_f = self._chunk_indices[cache_start_chunk_idx]
            rope_end_f = end_f

        local_data_info = copy.deepcopy(self._data_info)
        local_data_info["image_vae_embeds"] = image_vae_embeds

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred, _ = self._forward(
                latent_model_input,
                timestep,
                prompt_embeds,
                start_f=rope_start_f,
                end_f=rope_end_f,
                frame_index=frame_index,
                save_kv_cache=False,
                kv_cache=chunk_kv_cache,
                mask=self.mask,
                data_info=local_data_info,
                **self.model_kwargs,
            )
            if isinstance(noise_pred, Transformer2DModelOutput):
                noise_pred = noise_pred[0]
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)
            latents_dtype = latents.dtype
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        # Cache-building forward at t=0 so later chunks can attend to this one
        # (mirrors SANAStreamingSampler.sample with efficient_cache=False).
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        timestep = torch.zeros(latent_model_input.shape[0], device=device)
        _, updated_kv_cache = self._forward(
            latent_model_input,
            timestep,
            prompt_embeds,
            start_f=rope_start_f,
            end_f=rope_end_f,
            frame_index=frame_index,
            save_kv_cache=True,
            kv_cache=chunk_kv_cache,
            mask=self.mask,
            data_info=local_data_info,
            **self.model_kwargs,
        )
        self._kv_cache[chunk_idx] = updated_kv_cache
        self._promote_fixed_rope_full_history_cache(self._kv_cache, chunk_idx)

        return latents


class SANAStreamingPipeline:
    def __init__(self, device):
        self.device = device
        self.prompt = DEFAULT_PROMPT
        self.prompt_embeds = None
        self.prompt_mask = None
        self.negative_embeds = None
        self.negative_mask = None
        self.vae = None
        self.tokenizer = None
        self.text_encoder = None
        self.model = None
        self.config = None
        self.vae_dtype = None
        self.weight_dtype = None
        self.vae_stride = None
        self.latent_h = None
        self.latent_w = None
        self.flow_shift = None
        # Pinned-memory staging buffers for uint8 frame I/O (created lazily).
        self._pinned_in = None  # [H, W, 3] uint8 double buffer for H2D uploads
        self._pinned_in_events = None
        self._pinned_in_idx = 0
        self._pinned_out = None  # [C, T, H, W] uint8 buffer for D2H readback
        self._load_model()

    def _load_model(self):
        """Load the SANA-Streaming model and components."""
        config = pyrallis.parse(
            config_class=StreamingInferenceConfig,
            config_path=str(CONFIG_PATH),
            args=[],
        )
        self.config = config

        self.weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.vae_dtype = get_weight_dtype(config.vae.weight_dtype)
        self.vae_stride = config.vae.vae_stride

        if FRAME_WIDTH % self.vae_stride[2] or FRAME_HEIGHT % self.vae_stride[1]:
            raise ValueError(
                f"FRAME_WIDTH ({FRAME_WIDTH}) and FRAME_HEIGHT ({FRAME_HEIGHT}) must be "
                f"divisible by the VAE spatial stride ({self.vae_stride[2]}x{self.vae_stride[1]})"
            )
        self.latent_h = FRAME_HEIGHT // self.vae_stride[1]
        self.latent_w = FRAME_WIDTH // self.vae_stride[2]

        self.flow_shift = (
            config.scheduler.inference_flow_shift
            if config.scheduler.inference_flow_shift is not None
            else config.scheduler.flow_shift
        )

        # The LTX2VAE_chunk_tile loader enables temporal chunk tiling itself.
        print("[load] Loading VAE (Lightricks/LTX-2)...", flush=True)
        self.vae = get_vae(
            config.vae.vae_type,
            config.vae.vae_pretrained,
            device=self.device,
            dtype=self.vae_dtype,
            config=config.vae,
        )
        self.vae.streaming_causal_encode = _trt_compile(
            self.vae.streaming_causal_encode, "vae", "vae-encode"
        )

        from diffusion.utils.import_utils import is_flash_attn_available, is_xformers_available

        print("[load] VAE loaded. Loading text encoder (gemma-2-2b-it)...", flush=True)
        # FlashAttention is the default text-encoder backend when the kernel
        # is installed (falling back to SDPA if unavailable or broken, e.g. on
        # a brand-new GPU arch). Override with TEXT_ENCODER_ATTN=eager|sdpa|
        # flash_attention_2; an explicit choice is used as-is, no fallback.
        text_encoder_attn = os.environ.get("TEXT_ENCODER_ATTN")
        if text_encoder_attn is None:
            text_encoder_attn = "flash_attention_2" if is_flash_attn_available() else "sdpa"
            print(
                f"[load] Text encoder attn_implementation={text_encoder_attn} "
                "(default; TEXT_ENCODER_ATTN overrides)",
                flush=True,
            )
            try:
                self.tokenizer, self.text_encoder = load_text_encoder(
                    config.text_encoder.text_encoder_name, self.device, text_encoder_attn
                )
            except Exception as e:
                if text_encoder_attn == "sdpa":
                    raise
                print(f"[load] {text_encoder_attn} failed ({e}); retrying with sdpa", flush=True)
                self.tokenizer, self.text_encoder = load_text_encoder(
                    config.text_encoder.text_encoder_name, self.device, "sdpa"
                )
        else:
            print(f"[load] Text encoder attn_implementation={text_encoder_attn}", flush=True)
            self.tokenizer, self.text_encoder = load_text_encoder(
                config.text_encoder.text_encoder_name, self.device, text_encoder_attn
            )
        self.text_encoder = _trt_compile(self.text_encoder, "text_encoder", "text-encoder")

        print("[load] Text encoder loaded. Building DiT model...", flush=True)
        latent_size = config.model.image_size // config.vae.vae_downsample_rate
        model_kwargs = model_v2v_init_config(config, latent_size=latent_size)
        self.model = build_model(
            config.model.model,
            use_fp32_attention=config.model.get("fp32_attention", False),
            **model_kwargs,
        ).to(self.device)

        xformers_on = is_xformers_available() and os.environ.get("DISABLE_XFORMERS", "0") != "1"
        print(
            f"[load] DiT built (flash_attn={is_flash_attn_available()}, "
            f"xformers={xformers_on}, "
            f"chunkwise_gdn={os.environ.get('USE_CHUNKWISE_GDN', '1')}). "
            "Downloading/loading checkpoint...",
            flush=True,
        )
        model_path = f"hf://{STREAMING_REPO_ID}/dit/sana_streaming_ar.pth"
        state_dict = find_model(model_path)
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        if "state_dict" not in state_dict:
            state_dict = {
                "state_dict": {
                    key.removeprefix("model.").removeprefix("module."): value
                    for key, value in state_dict.items()
                }
            }
        state_dict = state_dict["state_dict"]
        if "pos_embed" not in state_dict and "pos_embed" in self.model.state_dict():
            state_dict["pos_embed"] = self.model.state_dict()["pos_embed"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.weight_dtype)

        if TORCHAO_QUANT:
            _quantize_dit_microscaling(self.model)

        print("[load] Weights loaded. Encoding prompt (first GPU compute)...", flush=True)
        self.set_prompt(self.prompt)

        print("SANA-Streaming model loaded successfully.", flush=True)

    def _encode_prompt(self, prompt: str, use_chi_prompt: bool = True):
        """Encode a text prompt, mirroring the reference inference script."""
        config = self.config
        max_length = config.text_encoder.model_max_length

        if use_chi_prompt:
            chi_prompt = "\n".join(config.text_encoder.chi_prompt)
            full_prompt = chi_prompt + prompt
            max_length = len(self.tokenizer.encode(chi_prompt)) + max_length - 2
        else:
            full_prompt = prompt

        tokens = self.tokenizer(
            full_prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            hidden_states = self.text_encoder(tokens.input_ids, tokens.attention_mask)[0]

        if use_chi_prompt:
            select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
            hidden_states = hidden_states[:, select_index]
            attention_mask = tokens.attention_mask[:, select_index]
        else:
            attention_mask = tokens.attention_mask

        return hidden_states[:, None], attention_mask

    def set_prompt(self, prompt: str):
        """Set the processing prompt for the streaming pipeline."""
        self.prompt = prompt
        self.prompt_embeds, self.prompt_mask = self._encode_prompt(prompt, use_chi_prompt=True)
        self.negative_embeds, self.negative_mask = self._encode_prompt("", use_chi_prompt=False)

    def preprocess_frame(self, image: Image.Image) -> torch.Tensor:
        """Resize+center-crop to the 720p bucket and normalize to [-1, 1].

        Uploads the frame to the GPU as raw uint8 (1 byte/channel instead of
        float32's 4 — 75% less PCIe traffic) through a pinned-memory double
        buffer so the DMA transfer runs asynchronously, then casts/normalizes
        to bfloat16 on the GPU before the Causal VAE encoder.
        """
        image = ImageOps.fit(image.convert("RGB"), (FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.BICUBIC)
        if self.device.type != "cuda":
            tensor = T.functional.to_tensor(image)
            tensor = T.functional.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            return tensor.to(self.device)
        if self._pinned_in is None:
            self._pinned_in = [
                torch.empty(FRAME_HEIGHT, FRAME_WIDTH, 3, dtype=torch.uint8, pin_memory=True)
                for _ in range(2)
            ]
            self._pinned_in_events = [torch.cuda.Event() for _ in range(2)]
        idx = self._pinned_in_idx
        self._pinned_in_idx ^= 1
        buf = self._pinned_in[idx]
        # Don't overwrite the buffer until a previous async H2D from it finished.
        self._pinned_in_events[idx].synchronize()
        buf.copy_(torch.from_numpy(np.array(image)))
        frame = buf.to(self.device, non_blocking=True).permute(2, 0, 1)
        self._pinned_in_events[idx].record()
        return frame.to(torch.bfloat16) / 127.5 - 1.0

    def vae_encode_stream(self, video_chunk: torch.Tensor, reset_cache: bool) -> torch.Tensor:
        """Causal streaming VAE encode with the normalization vae_encode() applies.

        Args:
            video_chunk: ``[1, C, T, H, W]`` pixel frames in [-1, 1].
            reset_cache: True for the first chunk of a stream.
        """
        vae = self.vae
        posterior = vae.streaming_causal_encode(
            video_chunk.to(dtype=vae.dtype), causal=True, reset_cache=reset_cache
        ).latent_dist
        z = posterior.mode()
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        return ((z - latents_mean) * vae.config.scaling_factor / latents_std).to(self.vae_dtype)

    def pixels_to_jpegs(self, video: torch.Tensor) -> list[bytes]:
        """Convert a decoded ``[B, C, T, H, W]`` tensor in [-1, 1] to JPEG frames.

        Downcasts to uint8 on the GPU before the D2H readback (1 byte/channel —
        75% less PCIe traffic than float32, 50% less than bf16), staged through
        a pinned-memory buffer so the copy is a true async DMA transfer.
        """
        video_u8 = ((video[0].float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        if video_u8.device.type == "cuda":
            if self._pinned_out is None or self._pinned_out.shape != video_u8.shape:
                self._pinned_out = torch.empty(video_u8.shape, dtype=torch.uint8, pin_memory=True)
            self._pinned_out.copy_(video_u8, non_blocking=True)
            # The JPEG encode below reads the buffer, so this D2H must complete.
            torch.cuda.current_stream().synchronize()
            video_u8 = self._pinned_out
        else:
            video_u8 = video_u8.cpu()
        frames = []
        for t in range(video_u8.shape[1]):
            img = T.functional.to_pil_image(video_u8[:, t])
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            frames.append(buf.getvalue())
        return frames


class VaeStreamingDecoder:
    """Chunk-by-chunk VAE decode with a persistent per-layer feature cache.

    Uses the library's causal streaming decoder when the decoder is causal and
    falls back to ``decode_with_cache`` (left-context prepend) otherwise.
    """

    def __init__(self, vae):
        self.vae = vae
        self.fresh = True
        if getattr(vae.decoder, "is_causal", False):
            self._causal_decoder = CausalVaeStreamingDecoder(vae)
        else:
            self._causal_decoder = None
        self.decode_chunk = _trt_compile(self.decode_chunk, "vae", "vae-decode")

    @torch.no_grad()
    def decode_chunk(self, z: torch.Tensor) -> torch.Tensor:
        """Decode one normalized latent chunk ``[B, C, F, H, W]`` to pixels in [-1, 1]."""
        if self._causal_decoder is not None:
            return self._causal_decoder.decode_chunk(z)
        vae = self.vae
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        z = (z * latents_std / vae.config.scaling_factor + latents_mean).to(vae.dtype)
        decoded = vae.decode_with_cache(
            z, temb=None, causal=False, return_dict=False, reset_cache=self.fresh
        )[0]
        self.fresh = False
        return decoded


def _make_placeholder_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT)).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


PLACEHOLDER_JPEG = _make_placeholder_jpeg()


class StreamSession:
    """State for one producer's stream: input buffer, persistent sampler/VAE
    caches, and a queue of processed output frames."""

    def __init__(self, pipeline: SANAStreamingPipeline, seed: int = 0):
        self.pipeline = pipeline
        p = pipeline
        self.latents_per_chunk = 24 // p.vae_stride[0]  # base_chunk_frames (3 for stride 8)
        self.frames_per_latent = p.vae_stride[0]  # pixel frames per latent after the first
        self.sampler = IncrementalStreamingSampler(
            p.model,
            condition=p.prompt_embeds,
            uncondition=p.negative_embeds,
            cfg_scale=1.0,  # long_streaming mode runs without CFG
            flow_shift=p.flow_shift,
            model_kwargs={"mask": p.prompt_mask},
            base_chunk_frames=self.latents_per_chunk,
            num_cached_blocks=2,
            cache_strategy="fixed_rope",
            efficient_cache=False,
            sink_token=True,
            autocast_dtype=p.weight_dtype,
        )
        self.generator = torch.Generator(device=p.device).manual_seed(seed)
        self.frame_buffer = []  # pixel tensors [C, H, W] in [-1, 1], on device
        self.source_latents = None  # encoded source latents not yet denoised
        self.output_queue = deque()  # processed JPEG bytes ready to send
        self.last_output = None
        self.decoder = VaeStreamingDecoder(p.vae)
        self.encoder_fresh = True  # reset the VAE encoder cache on next encode
        self.stream_started = False
        self.frames_seen = 0  # input frames received, for FRAME_SKIP pacing
        self._chunk_times = deque(maxlen=20)  # denoise+decode wall time per chunk

    def drop_oldest_if_overwhelmed(self) -> int:
        excess = len(self.frame_buffer) - MAX_PENDING_FRAMES
        if excess > 0:
            del self.frame_buffer[:excess]
        return excess

    @torch.no_grad()
    def advance(self):
        """Run the pipeline forward as far as the buffered input allows."""
        p = self.pipeline

        # VAE encode: the first pixel frame yields the first latent, then each
        # group of `frames_per_latent` frames yields one more latent frame.
        while True:
            needed = 1 if self.encoder_fresh else self.frames_per_latent
            if len(self.frame_buffer) < needed:
                break
            video_chunk = torch.stack(self.frame_buffer[:needed], dim=1).unsqueeze(0)  # [1, C, T, H, W]
            del self.frame_buffer[:needed]
            latent = p.vae_encode_stream(video_chunk, reset_cache=self.encoder_fresh)
            self.encoder_fresh = False
            self.source_latents = (
                latent if self.source_latents is None else torch.cat([self.source_latents, latent], dim=2)
            )

        # DiT denoise: the first chunk needs base_chunk_frames + 1 latents (the
        # sampler's minimum); later chunks are base_chunk_frames latents each.
        while self.source_latents is not None:
            needed = self.latents_per_chunk if self.stream_started else self.latents_per_chunk + 1
            if self.source_latents.shape[2] < needed:
                break
            chunk = self.source_latents[:, :, :needed]
            rest = self.source_latents[:, :, needed:]
            self.source_latents = rest if rest.shape[2] > 0 else None

            if not self.stream_started:
                print(
                    "[stream] denoising first chunk (Triton kernels compile here; "
                    "can take several minutes on a cold cache)...",
                    flush=True,
                )
                t0 = time.time()

            noise = torch.randn(
                1,
                p.config.vae.vae_latent_dim,
                needed,
                p.latent_h,
                p.latent_w,
                device=p.device,
                generator=self.generator,
            )
            t_chunk = time.time()
            denoised = self.sampler.sample_chunk(noise, chunk, steps=DENOISE_STEPS)
            pixels = self.decoder.decode_chunk(denoised.to(p.vae_dtype))
            self._chunk_times.append(time.time() - t_chunk)
            if not self.stream_started:
                print(
                    f"[stream] first chunk denoised+decoded in {time.time() - t0:.1f}s; "
                    f"pixel min/max/mean: {pixels.min().item():.2f}/"
                    f"{pixels.max().item():.2f}/{pixels.mean().item():.2f}",
                    flush=True,
                )
            jpegs = p.pixels_to_jpegs(pixels)
            self.output_queue.extend(jpegs)
            if self.stream_started and len(self._chunk_times) == self._chunk_times.maxlen:
                avg = sum(self._chunk_times) / len(self._chunk_times)
                # Sustainable output fps: pixel frames per chunk / seconds per chunk.
                fps = len(jpegs) / avg
                print(
                    f"[stream] throughput: {avg * 1000:.0f} ms/chunk, ~{fps:.1f} fps sustainable "
                    f"(dropped frames mean input rate exceeds this)",
                    flush=True,
                )
                self._chunk_times.clear()
            self.stream_started = True

    def next_output(self) -> bytes:
        if self.output_queue:
            self.last_output = self.output_queue.popleft()
        # While the first chunk is still being buffered (or the GPU is ahead of
        # the sender), repeat the last frame / a placeholder so the client's
        # request-response pacing keeps moving.
        return self.last_output if self.last_output is not None else PLACEHOLDER_JPEG


class SanaBackend:
    """Server-facing wrapper: one pipeline plus the current stream session.

    Exposes the same interface as detect_pipeline.DetectorPipeline:
    ``process_frame(data) -> bytes``, ``reset()``, ``set_prompt(prompt)``,
    and a ``prompt`` property. All methods must run on the server's single
    GPU worker thread.
    """

    def __init__(self, device):
        self.pipeline = SANAStreamingPipeline(device)
        self.session: StreamSession | None = None

    @property
    def prompt(self) -> str:
        return self.pipeline.prompt

    def set_prompt(self, prompt: str):
        self.pipeline.set_prompt(prompt)
        if self.session is not None:
            self.session.sampler.condition = self.pipeline.prompt_embeds
            self.session.sampler.mask = self.pipeline.prompt_mask

    def reset(self):
        self.session = None  # drops the DiT KV cache; VAE caches reset on next use

    def process_frame(self, data: bytes) -> bytes:
        """Decode one input frame, advance the pipeline, return one output frame."""
        if self.session is None:
            self.session = StreamSession(self.pipeline)
            warmup = 1 + self.session.latents_per_chunk * self.session.frames_per_latent
            print(
                f"[stream] new session; first real output after ~{warmup} input frames",
                flush=True,
            )
        session = self.session
        session.frames_seen += 1
        if session.frames_seen % 30 == 0:
            print(
                f"[stream] frames received: {session.frames_seen}, "
                f"buffered: {len(session.frame_buffer)}, "
                f"outputs pending: {len(session.output_queue)}",
                flush=True,
            )
        if (session.frames_seen - 1) % FRAME_SKIP == 0:
            frame = self.pipeline.preprocess_frame(Image.open(io.BytesIO(data)))
            session.frame_buffer.append(frame)
            dropped = session.drop_oldest_if_overwhelmed()
            if dropped:
                print(f"Backpressure: dropped {dropped} stale input frame(s)")
            session.advance()
        return session.next_output()
