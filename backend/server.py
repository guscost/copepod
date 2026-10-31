"""SANA-Streaming video-to-video websocket server.

Streams webcam frames through the SANA-Streaming V2V model (default prompt:
replace logos/ads with black squares) and broadcasts the processed frames to
all connected clients.

Architecture notes:
  * One "producer" connection sends frames; all connections (including the
    producer) receive the processed stream. The VAE keeps per-layer streaming
    caches in process memory, so only a single producer can run safely.
  * All GPU work is serialized on a single worker thread (``gpu_worker``) so
    the model, the VAE caches, and the DiT KV cache are never touched
    concurrently.
  * The DiT sampler keeps its autoregressive KV cache alive across calls
    (``IncrementalStreamingSampler``), which is what makes the output
    temporally consistent instead of frame-by-frame flicker.
"""

import asyncio
import copy
import io
import os
import sys
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# SANA-Streaming environment setup (must run before importing torch/diffusion).
os.environ["DISABLE_XFORMERS"] = "1"
os.environ.setdefault("USE_CHUNKWISE_GDN", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BACKEND_ROOT = Path(__file__).resolve().parent
# The Sana repo is not tracked in this repository; it must exist locally.
# Override with the SANA_ROOT environment variable if it lives elsewhere.
SANA_ROOT = Path(os.environ.get("SANA_ROOT", BACKEND_ROOT / "Sana-main")).resolve()
if (SANA_ROOT / "diffusion").is_dir() and str(SANA_ROOT) not in sys.path:
    sys.path.insert(0, str(SANA_ROOT))

try:
    import torch
    import torchvision.transforms as T
    from PIL import Image, ImageOps

    import diffusion.model.nets  # noqa: F401  (registers model/attention classes)
    import pyrallis
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae
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

STREAMING_REPO_ID = "Efficient-Large-Model/SANA-Streaming"
DEFAULT_PROMPT = "Replace all corporate logos, and all still or video advertising, with black squares"
CONFIG_PATH = BACKEND_ROOT / "sana_streaming_2b_720p.yaml"

# 720p training bucket; both dims must be divisible by the VAE spatial stride
# (32), which 1280x720 is not.
FRAME_WIDTH, FRAME_HEIGHT = 1280, 704
DENOISE_STEPS = 4  # long_streaming mode default
JPEG_QUALITY = 80
# If the GPU falls behind by more than this many unprocessed input frames,
# drop the oldest ones (a jump cut) instead of growing latency without bound.
MAX_PENDING_FRAMES = 48


@dataclass
class StreamingInferenceConfig:
    model: ModelVideoConfig
    vae: AEConfig
    text_encoder: TextEncoderConfig
    scheduler: SchedulerConfig


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

        self.latent_h = FRAME_HEIGHT // self.vae_stride[1]
        self.latent_w = FRAME_WIDTH // self.vae_stride[2]

        self.flow_shift = (
            config.scheduler.inference_flow_shift
            if config.scheduler.inference_flow_shift is not None
            else config.scheduler.flow_shift
        )

        # The LTX2VAE_chunk_tile loader enables temporal chunk tiling itself.
        self.vae = get_vae(
            config.vae.vae_type,
            config.vae.vae_pretrained,
            device=self.device,
            dtype=self.vae_dtype,
            config=config.vae,
        )

        self.tokenizer, self.text_encoder = get_tokenizer_and_text_encoder(
            config.text_encoder.text_encoder_name,
            device=self.device,
        )

        latent_size = config.model.image_size // config.vae.vae_downsample_rate
        model_kwargs = model_v2v_init_config(config, latent_size=latent_size)
        self.model = build_model(
            config.model.model,
            use_fp32_attention=config.model.get("fp32_attention", False),
            **model_kwargs,
        ).to(self.device)

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

        self.set_prompt(self.prompt)

        print("SANA-Streaming model loaded successfully.")

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
        """Resize+center-crop to the 720p bucket and normalize to [-1, 1]."""
        image = ImageOps.fit(image.convert("RGB"), (FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.BICUBIC)
        tensor = T.functional.to_tensor(image)
        tensor = T.functional.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return tensor.to(self.device)

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

    @staticmethod
    def pixels_to_jpegs(video: torch.Tensor) -> list[bytes]:
        """Convert a decoded ``[B, C, T, H, W]`` tensor in [-1, 1] to JPEG frames."""
        video = torch.clamp(127.5 * video[0].float() + 127.5, 0, 255).to(torch.uint8).cpu()
        frames = []
        for t in range(video.shape[1]):
            img = T.functional.to_pil_image(video[:, t])
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

            noise = torch.randn(
                1,
                p.config.vae.vae_latent_dim,
                needed,
                p.latent_h,
                p.latent_w,
                device=p.device,
                generator=self.generator,
            )
            denoised = self.sampler.sample_chunk(noise, chunk, steps=DENOISE_STEPS)
            pixels = self.decoder.decode_chunk(denoised.to(p.vae_dtype))
            self.output_queue.extend(p.pixels_to_jpegs(pixels))
            self.stream_started = True

    def next_output(self) -> bytes:
        if self.output_queue:
            self.last_output = self.output_queue.popleft()
        # While the first chunk is still being buffered (or the GPU is ahead of
        # the sender), repeat the last frame / a placeholder so the client's
        # request-response pacing keeps moving.
        return self.last_output if self.last_output is not None else PLACEHOLDER_JPEG


app = FastAPI(title="SANA-Streaming Video Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # credentials are invalid with a wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading SANA-Streaming pipeline on {device}...")
pipeline = SANAStreamingPipeline(device)

# All GPU work (and all session state) is serialized through this executor.
gpu_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sana-gpu")

connected_clients: set[WebSocket] = set()
producer: WebSocket | None = None
session: StreamSession | None = None


def _make_placeholder_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT)).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


PLACEHOLDER_JPEG = _make_placeholder_jpeg()


def _ingest_frame(data: bytes) -> bytes:
    """Decode one input frame, advance the pipeline, return one output frame.

    Runs exclusively on the GPU worker thread.
    """
    global session
    if session is None:
        session = StreamSession(pipeline)
    frame = pipeline.preprocess_frame(Image.open(io.BytesIO(data)))
    session.frame_buffer.append(frame)
    dropped = session.drop_oldest_if_overwhelmed()
    if dropped:
        print(f"Backpressure: dropped {dropped} stale input frame(s)")
    session.advance()
    return session.next_output()


def _reset_session():
    global session
    session = None  # drops the DiT KV cache; VAE caches reset on next use


def _apply_prompt(prompt: str):
    pipeline.set_prompt(prompt)
    if session is not None:
        session.sampler.condition = pipeline.prompt_embeds
        session.sampler.mask = pipeline.prompt_mask


async def broadcast(payload: bytes):
    stale = []
    for client in list(connected_clients):
        try:
            await client.send_bytes(payload)
        except Exception:
            stale.append(client)
    for client in stale:
        connected_clients.discard(client)


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    global producer
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"Client connected. Total clients: {len(connected_clients)}")

    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await websocket.receive_bytes()

            if producer is None:
                producer = websocket
                print("Producer connected.")
            if websocket is not producer:
                continue  # viewer connection: ignore inbound frames

            try:
                output = await loop.run_in_executor(gpu_worker, _ingest_frame, data)
            except Exception:
                traceback.print_exc()
                await websocket.close(code=1011)
                break

            await broadcast(output)

    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        connected_clients.discard(websocket)
        if websocket is producer:
            producer = None
            await loop.run_in_executor(gpu_worker, _reset_session)
            print("Producer disconnected; stream state reset.")
        print(f"Client removed. Total clients: {len(connected_clients)}")


class PromptUpdate(BaseModel):
    prompt: str


@app.post("/api/prompt")
async def set_prompt(update: PromptUpdate):
    """Set the processing prompt for the streaming pipeline."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(gpu_worker, _apply_prompt, update.prompt)
    return {"status": "ok", "prompt": update.prompt}


@app.get("/api/prompt")
async def get_prompt():
    """Get the current processing prompt."""
    return {"prompt": pipeline.prompt}


@app.post("/api/reset")
async def reset_state():
    """Reset the streaming state (DiT KV cache, VAE caches, frame buffers)."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(gpu_worker, _reset_session)
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "clients": len(connected_clients),
        "streaming": producer is not None,
    }


if __name__ == "__main__":
    import uvicorn

    # A single worker is required: the pipeline, VAE streaming caches, and the
    # DiT KV cache all live in process memory and are shared by connections.
    uvicorn.run(app, host="0.0.0.0", port=8000)
