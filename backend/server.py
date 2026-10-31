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

# SANA-Streaming environment setup
os.environ["DISABLE_XFORMERS"] = "1"
os.environ.setdefault("USE_CHUNKWISE_GDN", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from diffusion import DPMS
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.config import model_v2v_init_config
from diffusion.scheduler.sana_streaming_sampler import SANAStreamingSampler
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
        self.prompt_embeds = None
        self.prompt_mask = None
        self.negative_embeds = None
        self.negative_mask = None
        self.sampler = None
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
        import pyrallis
        
        # Load config
        config_path = "sana_streaming_2b_720p.yaml"
        config = pyrallis.parse(
            config_class=None,
            config_path=config_path,
            args=[]
        )
        self.config = config
        
        # Determine dtypes
        self.weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.vae_dtype = get_weight_dtype(config.vae.weight_dtype)
        self.vae_stride = config.vae.vae_stride
        
        # Latent dimensions for 720p frames (1280x720)
        self.latent_h = 720 // self.vae_stride[1]
        self.latent_w = 1280 // self.vae_stride[2]
        
        # Flow shift
        self.flow_shift = (
            config.scheduler.inference_flow_shift
            if config.scheduler.inference_flow_shift is not None
            else config.scheduler.flow_shift
        )
        
        # Load VAE
        self.vae = get_vae(
            config.vae.vae_type,
            config.vae.vae_pretrained,
            device=self.device,
            dtype=self.vae_dtype,
            config=config.vae
        )
        if config.vae.vae_type == "LTX2VAE_diffusers":
            if hasattr(self.vae, "enable_tiling"):
                self.vae.enable_tiling()
            if hasattr(self.vae, "use_framewise_encoding"):
                self.vae.use_framewise_encoding = True
                self.vae.use_framewise_decoding = True
                self.vae.tile_sample_stride_num_frames = getattr(config.vae, "tile_sample_stride_num_frames", 64)
                self.vae.tile_sample_min_num_frames = getattr(config.vae, "tile_sample_min_num_frames", 96)
        
        # Load tokenizer and text encoder
        self.tokenizer, self.text_encoder = get_tokenizer_and_text_encoder(
            config.text_encoder.text_encoder_name,
            device=self.device
        )
        
        # Load model
        latent_size = config.model.image_size // config.vae.vae_downsample_rate
        model_kwargs = model_v2v_init_config(config, latent_size=latent_size)
        self.model = build_model(
            config.model.model,
            use_fp32_attention=config.model.get("fp32_attention", False),
            **model_kwargs
        ).to(self.device)
        
        # Load weights
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
        
        # Encode default prompt
        self._encode_prompt(self.prompt)
        
        print("SANA-Streaming model loaded successfully.")
    
    def _encode_prompt(self, prompt: str):
        """Encode text prompt to embedding with chi_prompt support."""
        config = self.config
        max_length = config.text_encoder.model_max_length
        use_chi_prompt = True
        
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
        
        self.prompt_embeds = hidden_states[:, None]
        self.prompt_mask = attention_mask
        
        # Encode negative prompt
        negative_prompt = ""
        neg_tokens = self.tokenizer(
            negative_prompt,
            max_length=config.text_encoder.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            neg_hidden = self.text_encoder(neg_tokens.input_ids, neg_tokens.attention_mask)[0]
        
        self.negative_embeds = neg_hidden[:, None]
        self.negative_mask = neg_tokens.attention_mask
    
    def set_prompt(self, prompt: str):
        """Set the processing prompt for the streaming pipeline."""
        self.prompt = prompt
        self._encode_prompt(prompt)
    
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
            source_latent = vae_encode(
                self.config.vae.vae_type,
                self.vae,
                video_tensor,
                sample_posterior=False,
                device=self.device
            ).to(self.vae_dtype)
        
        # Create sampler for this frame
        base_chunk_frames = 24 // self.vae_stride[0]
        sampler = SANAStreamingSampler(
            self.model,
            condition=self.prompt_embeds,
            uncondition=self.negative_embeds,
            cfg_scale=1.0,
            flow_shift=self.flow_shift,
            model_kwargs={
                "data_info": {"image_vae_embeds": source_latent},
                "mask": self.prompt_mask
            },
            base_chunk_frames=base_chunk_frames,
            num_cached_blocks=2,
            cache_strategy="fixed_rope",
            efficient_cache=False,
            sink_token=True,
        )
        
        # Generate noise
        latent_t = source_latent.shape[2]
        generator = torch.Generator(device=self.device).manual_seed(0)
        noise = torch.randn(
            1,
            self.config.vae.vae_latent_dim,
            latent_t,
            self.latent_h,
            self.latent_w,
            device=self.device,
            generator=generator,
        )
        
        # Run diffusion (4 steps for real-time)
        with torch.no_grad():
            output_latent = sampler.sample(noise, steps=4).to(self.vae_dtype)
        
        # Decode back to image
        with torch.no_grad():
            output_video = vae_decode(self.config.vae.vae_type, self.vae, output_latent)
        
        # Convert back to PIL image
        output_frame = output_video[0, 0].detach().cpu()
        output_frame = T.functional.normalize(output_frame, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        output_frame = T.functional.to_pil_image(output_frame)
        
        return output_frame
    
    def reset_state(self):
        """Reset streaming state for new video."""
        pass

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
            image = image.resize((1280, 720))
            
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
