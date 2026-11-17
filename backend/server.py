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
                output = await loop.run_in_executor(gpu_worker, backend.process_frame, data)
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
