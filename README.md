# Copepod

Real-time video-to-video processing over FastAPI WebSockets.

## Quickstart

For an RTX 5090 install the CUDA toolkit (13.2 is tested), torch, and everything up to `# YOLO mode` in backend/requirements.txt plus `pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu132` (match your CUDA version), then run:

```
DETECT_EVERY=2 YOLO_MODEL=IDEA-Research/grounding-dino-tiny DETECT_CONF=0.25 GDINO_TEXT_THRESHOLD=0.5 GDINO_TRT=1 python server.py
```

## Modes

Exposes two selectable backend patterns (`BACKEND_MODE` env var):

- **`detect`** (default): logo/ad detection + blanking — detects logos and
  advertising with a zero-shot detector (Grounding DINO by default, or the
  lighter/faster YOLO-World), tracks the boxes across frames, and composites a
  fill (black square by default) over just those regions. Untouched pixels
  pass through unmodified. This is the right tool when the goal is "find
  logos/ads and blank them": it runs at full webcam framerate with
  millisecond latency, which full-frame diffusion cannot sustain.
- **`sana`**: SANA-Streaming diffusion V2V — regenerates every frame from a
  text prompt (`BACKEND_MODE=sana`).

## Architecture

```
[Client Webcam] → [Canvas Capture] → [WebSocket] → [detect + blank] → [Processed Output]
                                                 ↘ [SANA-Streaming Pipeline] ↗   (BACKEND_MODE=sana)
```

## Setup

### Backend
The backend wraps the Sana repo, which is **not** tracked in this repository —
clone it into `backend/Sana-main` first (or set the `SANA_ROOT` env var to its
location):

```bash
cd backend
git clone <sana-repo-url> Sana-main   # if not already present

# Install the Sana environment (Python 3.11 + CUDA; installs torch 2.9,
# flash-linear-attention, flash-attn, etc.). See Sana-main/environment_setup.sh.
cd Sana-main
bash environment_setup.sh sana   # creates/uses a conda env named "sana"
cd ..

# Then install the server-only deps and run:
pip install -r requirements.txt
python server.py
```

Note: install Sana's dependencies *before* `requirements.txt`, and don't pin
torch/diffusers/transformers yourself — versions must match
`Sana-main/pyproject.toml`.

Code layout: `server.py` is just the websocket/API plumbing; the two backends
live in `sana_pipeline.py` (diffusion V2V) and `detect_pipeline.py` (detection
+ blanking), each exposing `process_frame` / `reset` / `set_prompt` / `prompt`.

### Detector backend (logo/ad blanking, default)

```bash
python server.py   # or explicitly: BACKEND_MODE=detect python server.py
```

Uses Grounding DINO (zero-shot — detection classes are plain text, no
training) by default; `IDEA-Research/grounding-dino-base` auto-downloads from
Hugging Face on first run, and runs at ~30+ fps under fp16 autocast on a
high-end GPU.
Detect-only deployments can skip the Sana repo entirely — just
`pip install torch` plus `requirements.txt`. Knobs:

```bash
# Switch detector (any name containing "grounding" → Grounding DINO,
# "world" → YOLO-World, anything else → fine-tuned YOLO checkpoint)
YOLO_MODEL=IDEA-Research/grounding-dino-tiny   # smaller/faster Grounding DINO
YOLO_MODEL=yolov8s-worldv2.pt                  # lightweight YOLO-World
YOLO_MODEL=/path/to/logos.pt                   # fine-tuned YOLO (uses its trained classes)

# Zero-shot classes (Grounding DINO / YOLO-World only)
DETECT_CLASSES="corporate logo,advertising banner,sponsor logo"

# Confidence threshold (default 0.15 — zero-shot scores run low); doubles as
# Grounding DINO's box threshold
DETECT_CONF=0.25

# Fill style for detected regions (default black)
FILL_MODE=blur     # black | white | blur | debug (debug draws boxes+labels, no fill)

# Detect every Nth frame, reuse boxes in between (default 1)
DETECT_EVERY=2
```

With a zero-shot model, the "Apply Prompt" box doubles as a class editor:
enter a comma-separated class list (e.g. `nike logo, adidas logo`) and click
Apply. Tune classes/confidence with `FILL_MODE=debug` first so you can see
exactly what the detector is finding.

#### Grounding DINO knobs (default detector)

Grounding DINO is a zero-shot transformer detector — considerably more
accurate than YOLO-World on ambiguous classes like "logo": fewer false
positives, tighter boxes. Extras beyond the shared knobs:

```bash
# Token-to-phrase matching threshold (default 0.25; DETECT_CONF is the box threshold)
GDINO_TEXT_THRESHOLD=0.3

# Force full fp32 instead of the default fp16 autocast on CUDA
GDINO_FP16=0

# Experimental: compile the detector with Torch-TensorRT (~2x over eager fp16
# once compiled). Needs a torch-tensorrt build matching your torch/CUDA (see
# requirements.txt). Compilation happens on the warmup frame at startup, so
# set FRAME_WIDTH/FRAME_HEIGHT to the actual incoming frame size (default
# 1280x720) — a different first-frame size triggers a slow recompile. A failed
# TensorRT call permanently falls back to eager rather than killing the stream.
GDINO_TRT=1 FRAME_WIDTH=1280 FRAME_HEIGHT=720 python server.py
```

### Reducing GPU load / power draw

The backend exposes env-var
knobs to lighten the per-frame and sustained load:

```bash
# 480p instead of 720p: ~2x less DiT/VAE work per chunk
FRAME_WIDTH=832 FRAME_HEIGHT=480 python server.py

# Half the denoising steps (2 instead of 4)
DENOISE_STEPS=2 python server.py

# Process every other input frame (halves sustained processing framerate)
FRAME_SKIP=2 python server.py
```

`FRAME_WIDTH`/`FRAME_HEIGHT` must both be divisible by 32. The knobs combine
freely.

### Frontend
Open `frontend/index.html` in a browser, or serve it:
```bash
cd frontend
python -m http.server 8080
```

## Usage

1. Start the backend server
2. Open the frontend in a browser
3. Click "Start Stream" to begin webcam capture
4. Enter a processing prompt and click "Apply Prompt"
5. Watch the real-time processed output

## Features

- Real-time video processing with sub-100ms latency
- Configurable processing prompts
- State caching for streaming consistency
- FPS monitoring
- Backpressure handling to prevent frame flooding
- Default prompt: "Replace all corporate logos, and all still or video advertising, with black squares"

## Model

Uses the SANA-Streaming 2B model from [Efficient-Large-Model/SANA-Streaming](https://huggingface.co/Efficient-Large-Model/SANA-Streaming) with:
- 4 denoising steps for real-time performance
- Chunk-wise generation with state caching
- Sink-token caching for long-range consistency

## Production Optimizations

Status of the usual inference optimizations in this stack:

- **FlashAttention** (default): the DiT softmax attention prefers flash-attn
  when installed, and the Gemma text encoder defaults to
  `flash_attention_2` (falls back to SDPA if the kernel is unavailable or
  fails to load). Override the text-encoder backend with
  `TEXT_ENCODER_ATTN=eager|sdpa|flash_attention_2`; force the DiT off flash
  with `DISABLE_FLASH_ATTN=1`. Note the DiT's flash path runs in bf16, so it
  bypasses the config's `fp32_attention: true` (which only applies to the
  GDN/SDPA paths).
- **xFormers** (default): installed by Sana's `environment_setup.sh`; drives
  the V2V cross-attention packing path and is the fallback when flash-attn
  is absent. Force off with `DISABLE_XFORMERS=1` (the previous behavior).
- **TensorRT** (experimental, opt-in): `TENSORRT=vae,text_encoder` (or
  `TENSORRT=all`) compiles those components with Torch-TensorRT on first
  use, with automatic permanent fallback to eager if a compiled call fails.
  Requires a torch-tensorrt build matching the installed torch/CUDA (see
  `backend/requirements.txt`). The DiT is not compilable this way — its
  stateful KV cache and Triton GDN kernels don't survive graph capture.
- **Quantization**: bf16 mixed precision (`mixed_precision: bf16` in
  `sana_streaming_2b_720p.yaml`); no fp8/int8 path for this pipeline.
- **Persistent model loading** (already in place): the backend loads once in
  the FastAPI lifespan and is reused across all connections.

Verify what's active from the startup logs: `[load] Text encoder
attn_implementation=...`, `[load] DiT built (flash_attn=..., xformers=...)`,
and `[tensorrt] ...` lines.
