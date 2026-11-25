"""Copepod streaming video server.

Receives webcam frames over a websocket, processes them with the selected
backend, and broadcasts the processed frames to all connected clients.

Backends (BACKEND_MODE env var), each exposing the same interface
(``process_frame(jpeg_bytes) -> jpeg_bytes``, ``reset()``,
``set_prompt(text)``, ``prompt``):

  * "detect" (default): logo/ad detection (Grounding DINO or YOLO)
    + blanking (detect_pipeline.py) — only alters detected regions; much faster and
    lower latency.
  * "sana": SANA-Streaming diffusion V2V (sana_pipeline.py) — regenerates
    every frame from a text prompt.

Architecture notes:
  * One "producer" connection sends frames; all connections (including the
    producer) receive the processed stream. Both backends keep per-stream
    state (VAE/DiT caches or tracker state) in process memory, so only a
    single producer can run safely.
  * All GPU work and all backend state is serialized on a single worker
    thread (``gpu_worker``).
"""

import asyncio
import os
import queue
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

try:
    import torch
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Failed to import server dependencies: {e}\n"
        "Install them with:  pip install -r backend/requirements.txt"
    ) from e

BACKEND_MODE = os.environ.get("BACKEND_MODE", "detect")  # sana|detect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = None  # SanaBackend | DetectorPipeline; created in lifespan startup

# All GPU work (and all backend state) is serialized through this executor.
gpu_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")

connected_clients: set[WebSocket] = set()
producer: WebSocket | None = None
h264_decoder = None
h264_output_encoder = None
STREAM_OUTPUT_FORMAT = os.environ.get("STREAM_OUTPUT_FORMAT", "h264").strip().lower()
if STREAM_OUTPUT_FORMAT not in ("h264", "jpeg", "webp"):
    raise SystemExit("Unknown STREAM_OUTPUT_FORMAT (expected 'h264', 'jpeg', or 'webp')")


class H264OutputEncoder:
    """Encode processed JPEG frames as a low-latency fragmented H264 MP4 stream."""

    def __init__(self, loop, fps: int = 24):
        self.loop = loop
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-f", "image2pipe",
                "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
                "-profile:v", "baseline", "-level", "3.1",
                "-g", str(fps), "-keyint_min", str(fps), "-bf", "0",
                "-movflags", "frag_keyframe+empty_moov+default_base_moof",
                "-f", "mp4", "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        while self.process.poll() is None:
            chunk = self.process.stdout.read(65536)
            if not chunk:
                break
            asyncio.run_coroutine_threadsafe(broadcast(chunk), self.loop)

    def encode(self, jpeg_bytes: bytes):
        try:
            self.process.stdin.write(jpeg_bytes)
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()


class H264FrameDecoder:
    """Decode Annex-B H264 chunks from WebCodecs into JPEG frames."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.frames = queue.Queue(maxsize=4)
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-fflags", "nobuffer",
                "-flags", "low_delay", "-f", "h264", "-i", "pipe:0",
                "-an", "-vsync", "0", "-f", "image2pipe", "-vcodec", "mjpeg",
                "-q:v", "2", "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.thread = threading.Thread(target=self._read_output, daemon=True)
        self.thread.start()

    def _read_output(self):
        pending = bytearray()
        while self.process.poll() is None:
            chunk = self.process.stdout.read(65536)
            if not chunk:
                break
            pending.extend(chunk)
            while True:
                start = pending.find(b"\xff\xd8")
                if start < 0:
                    break
                end = pending.find(b"\xff\xd9", start + 2)
                if end < 0:
                    if len(pending) > 8 * 1024 * 1024:
                        pending.clear()
                    break
                frame = bytes(pending[start:end + 2])
                del pending[:end + 2]
                try:
                    self.frames.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass
                    self.frames.put_nowait(frame)

    def decode(self, chunk: bytes, timeout: float = 0.25):
        if not chunk:
            return None
        try:
            self.process.stdin.write(chunk)
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError):
            return None
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load at server startup, not at import time, so diagnostic tooling can
    # import this module without touching the GPU. The backend import is
    # lazy so detect mode never needs the Sana repo (and vice versa).
    global backend
    if BACKEND_MODE == "detect":
        from detect_pipeline import DetectorPipeline

        print(f"Loading object-detector backend on {device}...", flush=True)
        backend = DetectorPipeline(device)
    elif BACKEND_MODE == "sana":
        from sana_pipeline import SanaBackend

        print(f"Loading SANA-Streaming backend on {device}...", flush=True)
        backend = SanaBackend(device)
    else:
        raise SystemExit(f"Unknown BACKEND_MODE {BACKEND_MODE!r} (expected 'sana' or 'detect')")
    yield


app = FastAPI(title="Copepod Video Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # credentials are invalid with a wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    global producer, h264_decoder, h264_output_encoder
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"Client connected. Total clients: {len(connected_clients)}")

    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await websocket.receive_bytes()

            if producer is None:
                producer = websocket
                h264_decoder = H264FrameDecoder()
                requested_output = websocket.query_params.get("output", "h264")
                if STREAM_OUTPUT_FORMAT == "h264" and requested_output == "h264":
                    h264_output_encoder = H264OutputEncoder(loop)
                print("Producer connected.")
            if websocket is not producer:
                continue  # viewer connection: ignore inbound frames

            if data[:1] == b"H":
                data = h264_decoder.decode(data[1:])
                if data is None:
                    continue

            try:
                output = await loop.run_in_executor(gpu_worker, backend.process_frame, data)
            except Exception:
                traceback.print_exc()
                await websocket.close(code=1011)
                break

            if h264_output_encoder is not None:
                h264_output_encoder.encode(output)
            else:
                await broadcast(output)

    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        connected_clients.discard(websocket)
        if websocket is producer:
            producer = None
            if h264_decoder is not None:
                h264_decoder.close()
                h264_decoder = None
            if h264_output_encoder is not None:
                h264_output_encoder.close()
                h264_output_encoder = None
            await loop.run_in_executor(gpu_worker, backend.reset)
            print("Producer disconnected; stream state reset.")
        print(f"Client removed. Total clients: {len(connected_clients)}")


class PromptUpdate(BaseModel):
    prompt: str


@app.post("/api/prompt")
async def set_prompt(update: PromptUpdate):
    """Set the processing prompt (detection classes in detect mode)."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(gpu_worker, backend.set_prompt, update.prompt)
    return {"status": "ok", "prompt": update.prompt}


@app.get("/api/prompt")
async def get_prompt():
    """Get the current processing prompt (detection classes in detect mode)."""
    return {"prompt": backend.prompt}


@app.post("/api/reset")
async def reset_state():
    """Reset the per-stream state (DiT/VAE caches or tracker state)."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(gpu_worker, backend.reset)
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "mode": BACKEND_MODE,
        "device": str(device),
        "clients": len(connected_clients),
        "streaming": producer is not None,
    }


if __name__ == "__main__":
    import uvicorn

    # A single worker is required: the backend and its per-stream state live
    # in process memory and are shared by connections.
    uvicorn.run(app, host="0.0.0.0", port=8000)
