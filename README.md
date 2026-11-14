# Copepod

Real-time video-to-video style transformation using SANA-Streaming with FastAPI WebSockets.

## Architecture

```
[Client Webcam] → [Canvas Capture] → [WebSocket] → [SANA-Streaming Pipeline] → [Processed Output]
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

### Reducing GPU load / power draw

If the machine crashes or reboots when processing starts (a classic symptom of
transient power spikes, e.g. on an RTX 5090), the backend exposes env-var
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
freely. The most direct fix for power-spike crashes, though, is capping the
card's power limit, e.g. `nvidia-smi -pl 400` — often sufficient on its own.

To confirm which stage over-stresses the machine, run
`python diagnose_gpu.py` and watch `nvidia-smi` power draw in parallel.

### If it still crashes with power/clock limits applied

A whole-machine crash that persists even with `nvidia-smi -pl` and locked
clocks is usually **not** power delivery — it's a wedged CUDA/Triton kernel
hanging the GPU (a known failure mode on new architectures like the RTX
5090's Blackwell/sm_120). Run the diagnostic with synchronous kernel launches
so the last printed stage is the true culprit:

```bash
CUDA_LAUNCH_BLOCKING=1 python diagnose_gpu.py
```

Match the crash point to the culprit:

- **Crash right after `[load] Weights loaded. Encoding prompt`**: the Gemma
  text encoder's first forward pass (Stage 2b in the diagnostic). No Triton
  kernels are involved at that point — suspects are the attention backend and
  cuBLAS bf16 GEMMs. Try forcing the encoder's attention implementation:

  ```bash
  TEXT_ENCODER_ATTN=eager python diagnose_gpu.py   # or: python server.py
  ```

  (`eager` = plain PyTorch matmul attention; `sdpa` and `flash_attention_2`
  are the other choices.)

- **Crash in "Stage 5" / on the first processed frame**: first A/B the
  flash-attn CUDA kernels (used by the DiT softmax attention layers) against
  PyTorch's SDPA fallback:

  ```bash
  DISABLE_FLASH_ATTN=1 python diagnose_gpu.py
  ```

  If the crash disappears, flash-attn was misbuilt for your GPU arch — keep
  `DISABLE_FLASH_ATTN=1` set when running `server.py` (slightly slower, but
  stable). If it still dies with flash-attn disabled, the fused GDN Triton
  kernels are at fault; that path has no PyTorch fallback in this repo, so
  the next step is a driver/triton/fla version bump or an upstream issue.
  If you see **"unspecified launch failure"** (cudaError 719): a kernel
  crashed in flight (GPU-side segfault). First clear the Triton kernel cache
  — cubins cached under a different GPU arch/driver cause exactly this:

  ```bash
  rm -rf ~/.triton/cache
  CUDA_LAUNCH_BLOCKING=1 TEXT_ENCODER_ATTN=eager DISABLE_FLASH_ATTN=1 python diagnose_gpu.py dit
  ```

  (`dit` mode skips everything except the Stage 5 DiT forward, for fast
  iteration.) If it still faults, `compute-sanitizer --tool memcheck python
  diagnose_gpu.py dit` will name the faulting kernel, and `TRITON_ALWAYS_COMPILE=1`
  forces recompilation at build time instead of launch.

Also check the kernel log from the crashed boot for the smoking gun:

```bash
journalctl -k -b -1 | grep -i xid
```

Xid 13/31 = a kernel faulted (software, fixable as above). Xid 79 = "GPU has
fallen off the bus" — a **hardware/link** event, not a software bug: the card
momentarily vanished from PCIe. On an eGPU (USB4/Thunderbolt) this means link
or power delivery, and CUDA errors like "unspecified launch failure" are just
the aftermath. Confirmed triggers/fixes, in order of likelihood:

1. **Enclosure PSU vs transients**: even with `nvidia-smi -pl`, the 5090's
   microsecond power spikes can dip the rail and drop the card. Lower the cap
   further (`-pl 300`) and lock clocks (`-lgc 1200`) to find the stable
   envelope; the `FRAME_WIDTH`/`FRAME_HEIGHT`/`DENOISE_STEPS`/`FRAME_SKIP`
   knobs above reduce burst current too.
2. **12V-2x6 connector**: reseat until it clicks; no pigtails/splitters.
3. **Cable**: short (≤0.5m) certified 40Gbps TB4/USB4 cable, direct to the
   port — no hubs or daisy chains.
4. **PCIe ASPM**: link power management is a classic eGPU dropper — boot
   with the `pcie_aspm=off` kernel parameter.
5. **Link health**: watch for correctable PCIe errors while streaming —
   `journalctl -k -f | grep -i aer`. A stream of AER errors before the drop
   means a marginal link (cable/port/thermal on the enclosure's TB
   controller).

Xid 74 = NVLink error (not applicable to PCIe cards).

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
