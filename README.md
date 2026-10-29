# Copepod

Real-time video-to-video style transformation using SANA-Streaming with FastAPI WebSockets.

## Architecture

```
[Client Webcam] → [Canvas Capture] → [WebSocket] → [SANA-Streaming Pipeline] → [Processed Output]
```

## Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
python server.py
```

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
