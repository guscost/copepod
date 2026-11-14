# Copepod

Real-time video-to-video processing over FastAPI WebSockets, with two
selectable backends (`BACKEND_MODE` env var):

- **`sana`** (default): SANA-Streaming diffusion V2V — regenerates every
  frame from a text prompt.
- **`detect`**: YOLO logo/ad detection + blanking — detects logos and
  advertising, tracks the boxes across frames, and composites a fill (black
  square by default) over just those regions. Untouched pixels pass through
  unmodified. This is the right tool when the goal is "find logos/ads and
  blank them": it runs at full webcam framerate with millisecond latency,
  which full-frame diffusion cannot sustain.

## Architecture

```
[Client Webcam] → [Canvas Capture] → [WebSocket] → [SANA-Streaming Pipeline] → [Processed Output]
                                                 ↘ [YOLO detect + blank] ↗   (BACKEND_MODE=detect)
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
live in `sana_pipeline.py` (diffusion V2V) and `detect_pipeline.py` (YOLO
blanking), each exposing `process_frame` / `reset` / `set_prompt` / `prompt`.

### Detector backend (logo/ad blanking)

```bash
BACKEND_MODE=detect python server.py
```

Uses YOLO-World (zero-shot — detection classes are plain text, no training)
by default; `yolov8s-worldv2.pt` auto-downloads on first run. Detect-only
deployments can skip the Sana repo entirely — just `pip install torch` plus
`requirements.txt`. Knobs:

```bash
# Point at a fine-tuned YOLO checkpoint instead (uses its trained classes)
YOLO_MODEL=/path/to/logos.pt

# Zero-shot classes (YOLO-World only)
DETECT_CLASSES="corporate logo,advertising banner,sponsor logo"

# Confidence threshold (default 0.15 — zero-shot scores run low)
DETECT_CONF=0.25

# Fill style for detected regions (default black)
FILL_MODE=blur     # black | white | blur | debug (debug draws boxes+labels, no fill)

# Detect every Nth frame, reuse boxes in between (default 1)
DETECT_EVERY=2
```

With the YOLO-World model, the "Apply Prompt" box doubles as a class editor:
enter a comma-separated class list (e.g. `nike logo, adidas logo`) and click
Apply. Tune classes/confidence with `FILL_MODE=debug` first so you can see
exactly what the detector is finding.

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

- Use TensorRT compilation for faster inference
- Enable xFormers/FlashAttention for memory efficiency
- Consider model quantization for lower VRAM usage
- Use persistent model loading to avoid reload overhead
