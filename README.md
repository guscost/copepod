# Copepod

Real-time video-to-video processing over FastAPI WebSockets.

## Quickstart

For the default ad-blocking fine-tuned YOLO detector, install the CUDA toolkit (13.2 is tested), torch, and the normal items in backend/requirements.txt, then in the backend folder run:

```
python server.py
```

For freeform detection, try the YOLOv8 World 2 model:

```
YOLO_MODEL=yolov8s-worldv2.pt python server.py
```

Open frontend/index.html, and start the webcam stream.

## Grounding DINO for better general detection

Add `pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu132` (match your CUDA version), then run e.g. (for a 5090):

```
DETECT_EVERY=4 YOLO_MODEL=IDEA-Research/grounding-dino-tiny DETECT_CONF=0.3 GDINO_TEXT_THRESHOLD=0.5 GDINO_TRT=1 python server.py
```

## Modes

Exposes two selectable backend patterns (`BACKEND_MODE` env var):

- **`detect`** (default): logo/ad detection + blanking — detects logos and
  advertising with two fine-tuned YOLO checkpoints combined by default (a
  billboard detector plus a brand-logo detector), tracks the boxes across
  frames, and composites a fill (black square by default) over just those
  regions. Untouched pixels pass through unmodified. This is the right tool
  when the goal is "find logos/ads and blank them": it runs at full webcam
  framerate with millisecond latency, which full-frame diffusion cannot
  sustain.
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

Runs two fine-tuned YOLO checkpoints together by default —
[`maco018/billboard-detection-Yolo11`](https://huggingface.co/maco018/billboard-detection-Yolo11)
(billboards) and
[`haydarkadioglu/brand-eye`](https://huggingface.co/haydarkadioglu/brand-eye)
(33 car-brand logos). Both auto-download from Hugging Face on first run, and
their detections are combined into one track set. Either can be disabled or
swapped, and a single `YOLO_MODEL` still selects any one detector (zero-shot
or fine-tuned) as before.
Detect-only deployments can skip the Sana repo entirely — just
`pip install torch` plus `requirements.txt`. Knobs:

```bash
# Default pair, individually replaceable (owner/repo[:file] on Hugging Face
# or a local .pt path) or disabled with "off"
BILLBOARD_YOLO=maco018/billboard-detection-Yolo11:yolo11m.pt   # n/s/m/l/x variants
BRAND_EYE_YOLO=off                                             # disable the brand-logo detector

# Single-detector override (any name containing "grounding" → Grounding DINO,
# "world" → YOLO-World, anything else → one fine-tuned YOLO checkpoint)
YOLO_MODEL=IDEA-Research/grounding-dino-base   # zero-shot Grounding DINO
YOLO_MODEL=yolov8s-worldv2.pt                  # lightweight YOLO-World
YOLO_MODEL=/path/to/logos.pt                   # fine-tuned YOLO (uses its trained classes)

# Zero-shot classes (Grounding DINO / YOLO-World only)
DETECT_CLASSES="corporate logo,advertising banner,sponsor logo"

# Confidence threshold (default 0.25; zero-shot scores run low — use ~0.15
# for Grounding DINO / YOLO-World); doubles as Grounding DINO's box threshold
DETECT_CONF=0.3

# Fill style for detected regions (default black)
FILL_MODE=blur     # black | white | blur | debug (debug draws boxes+labels, no fill)

# Detect every Nth frame, reuse boxes in between (default 1)
DETECT_EVERY=2

# Optional minimum size filters (disabled by default)
MIN_BOX_AREA=0.0001       # fraction of frame area
MIN_BOX_WIDTH=8           # pixels
MIN_BOX_HEIGHT=8          # pixels

# Compress output more aggressively (jpeg default 80)
JPEG_QUALITY=65
OUTPUT_FORMAT=jpeg        # jpeg (default) | webp
```

With a zero-shot model, the "Apply Prompt" box doubles as a class editor:
enter a comma-separated class list (e.g. `nike logo, adidas logo`) and click
Apply. Tune classes/confidence with `FILL_MODE=debug` first so you can see
exactly what the detector is finding.

#### Grounding DINO knobs

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

For the webcam-to-server side, the frontend defaults to 24 FPS and JPEG input
quality 0.55. These can be tuned in the URL, for example:

```
frontend/index.html?fps=24&quality=0.45
```

Lower `quality` reduces upload bandwidth; `fps` is capped at 60 and is
backpressure-controlled, so the browser does not enqueue a new frame while the
previous frame is still processing. Input frames default to WebP for a smaller
upload than JPEG:

```
frontend/index.html?fps=24&format=h264
```

WebP input is the default because direct WebSocket frame delivery has the
lowest latency and avoids two FFmpeg transcode stages. Use `format=h264` to
opt into WebCodecs upload and MediaSource output, or `output=h264` to keep
WebP/JPEG input with H264 output. H264 requires `ffmpeg` on the PATH and is
best used only when bandwidth reduction is more important than immediate
start and minimum latency. Set `STREAM_OUTPUT_FORMAT=webp` or `jpeg` to
disable H264 output independently.

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
- **SDPA over xFormers** (default): the V2V paths run on native PyTorch SDPA
  (FlashAttention backend, explicitly enabled) since the core kernels are the
  maintained path on Blackwell (sm_120). Re-enable xFormers — which drives the
  V2V cross-attention packing path — with `DISABLE_XFORMERS=0`.
- **TensorRT** (experimental, opt-in): `TENSORRT=vae,text_encoder` (or
  `TENSORRT=all`) compiles those components with Torch-TensorRT on first
  use, with automatic permanent fallback to eager if a compiled call fails.
  Uses the Blackwell dynamo settings: explicit typing, optimization level 4,
  CUDA graphs (cuts WSL 2/WDDM launch overhead), and a persistent engine
  cache in `~/.cache/sana_trt_engines` (Inductor/Triton caches live in
  `~/.cache/sana_inductor` / `~/.cache/sana_triton`). Requires a
  torch-tensorrt build matching the installed torch/CUDA (see
  `backend/requirements.txt`). The DiT is not compilable this way — its
  stateful KV cache and Triton GDN kernels don't survive graph capture.
- **TorchAO microscaling quantization** (experimental, opt-in): `TORCHAO_QUANT=1`
  quantizes the DiT with NVFP4 weight-only (block size 16) for bulk
  FFN-out/attention GEMMs and MXFP8 dynamic (block size 32) for the sensitive
  FFN fc1 and GDN linear layers; conditioning, embeddings, output heads, and
  norms stay in bf16. The NVFP4/MXFP8 configs come from torchao's *prototype*
  mx_formats API, so this needs a recent `torchao` and a Blackwell GPU
  (sm_100+/sm_120); older torchao or older GPUs degrade to int4 weight-only +
  int8 dynamic, and without torchao the DiT stays bf16. Baseline remains bf16
  mixed precision (`mixed_precision: bf16` in `sana_streaming_2b_720p.yaml`).
- **uint8 frame I/O** (default): frames cross the PCIe bus as uint8 both
  directions — webcam frames upload as 1 byte/channel and are
  cast/normalized to bf16 on the GPU before the Causal VAE encoder; VAE
  decoder output is downcast to uint8 on the GPU before the D2H readback.
  Both directions stage through pinned-memory buffers (double-buffered on
  input) so DMA transfers run asynchronously.
- **Persistent model loading** (already in place): the backend loads once in
  the FastAPI lifespan and is reused across all connections.

Verify what's active from the startup logs: `[load] Text encoder
attn_implementation=...`, `[load] DiT built (flash_attn=..., xformers=...)`,
and `[tensorrt] ...` lines.
