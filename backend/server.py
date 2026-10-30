import asyncio
import io
import os
import time
from typing import Optional

import torch
import torchvision.transforms as T
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# SANA-Streaming imports
os.environ["DISABLE_XFORMERS"] = "1"
os.environ.setdefault("USE_CHUNKWISE_GDN", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from diffusion import DPMSolverMultistepScheduler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.config import model_v2v_init_config
from tools.download import find_model

app = FastAPI(title="SANA-Streaming Video Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for the streaming pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading SANA-Streaming pipeline on {device}...")

STREAMING_REPO_ID = "Efficient-Large-Model/SANA-Streaming"
DEFAULT_PROMPT = "Replace all corporate logos, and all still or video advertising, with black squares"

# Connected clients for broadcast
connected_clients = []

class SANAStreamingPipeline:
    def __init__(self, device):
        self.device = device
        self.prompt = DEFAULT_PROMPT
        self.state = None  # State caching for streaming
        
        # Load model configuration
        config = model_v2v_init_config(
            "configs/sana_streaming/sana_streaming_2b_720p.yaml",
            model_path=f"hf://{STREAMING_REPO_ID}/dit/sana_streaming_ar.pth",
        )
        
        # Build model components
        self.model = build_model(config, device)
        self.vae = get_vae(config.vae, device)
        self.tokenizer, self.text_encoder = get_tokenizer_and_text_encoder(config.text_encoder, device)
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.scheduler.num_train_timesteps,
            beta_schedule=config.scheduler.beta_schedule,
            prediction_type=config.scheduler.prediction_type,
        )
        
        print("SANA-Streaming model loaded successfully.")
    
    def set_prompt(self, prompt: str):
        self.prompt = prompt
    
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embedding."""
        tokens = self.tokenizer(
            prompt,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            embed = self.text_encoder(**tokens).last_hidden_state
        return embed
    
    def process_frame(self, image: Image.Image) -> Image.Image:
        """Process a single frame using SANA-Streaming with state caching."""
        # Convert PIL image to tensor
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        video_tensor = transform(image).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, C, H, W]
        
        # Encode to latent space
        with torch.no_grad():
            source_latent = vae_encode(self.vae, video_tensor)
        
        # Encode prompt
        prompt_embed = self.encode_prompt(self.prompt)
        
        # Run diffusion with state caching (4 steps for real-time)
        with torch.no_grad():
            # Add noise
            noise = torch.randn_like(source_latent)
            t = torch.tensor([999], device=self.device)
            noisy_latent = self.scheduler.add_noise(source_latent, noise, t)
            
            # Denoise with state caching
            if self.state is None:
                self.state = {"step": 0, "cached_blocks": []}
            
            # Simplified single-step denoising for streaming
            # In production, this would use the full chunk-wise generation
            output_latent = self.model(
                noisy_latent,
                t,
                prompt_embed,
                source_latent,
                state=self.state,
            )
            
            # Decode back to image
            output_video = vae_decode(self.vae, output_latent)
        
        # Convert back to PIL image
        output_frame = output_video[0, 0].detach().cpu()
        output_frame = T.functional.normalize(output_frame, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        output_frame = T.functional.to_pil_image(output_frame)
        
        return output_frame
    
    def reset_state(self):
        """Reset streaming state for new video."""
        self.state = None

pipeline = SANAStreamingPipeline(device)

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"Client connected. Total clients: {len(connected_clients)}")
    
    try:
        while True:
            # Receive binary frame from client
            data = await websocket.receive_bytes()
            
            # Decode image
            image = Image.open(io.BytesIO(data)).convert("RGB")
            image = image.resize((512, 512))
            
            # Process with SANA-Streaming pipeline
            loop = asyncio.get_running_loop()
            processed = await loop.run_in_executor(
                None, pipeline.process_frame, image
            )
            
            # Encode processed frame
            img_byte_arr = io.BytesIO()
            processed.save(img_byte_arr, format='JPEG', quality=80)
            processed_bytes = img_byte_arr.getvalue()
            
            # Broadcast to all connected clients
            disconnected = []
            for client in connected_clients:
                try:
                    await client.send_bytes(processed_bytes)
                except Exception:
                    disconnected.append(client)
            
            # Clean up disconnected clients
            for client in disconnected:
                connected_clients.remove(client)
            
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        print(f"Client removed. Total clients: {len(connected_clients)}")

@app.post("/api/prompt")
async def set_prompt(prompt: str):
    """Set the processing prompt for the streaming pipeline."""
    pipeline.set_prompt(prompt)
    return {"status": "ok", "prompt": prompt}

@app.get("/api/prompt")
async def get_prompt():
    """Get the current processing prompt."""
    return {"prompt": pipeline.prompt}

@app.post("/api/reset")
async def reset_state():
    """Reset the streaming state."""
    pipeline.reset_state()
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
