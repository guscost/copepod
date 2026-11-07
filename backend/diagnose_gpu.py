"""Stage-by-stage GPU diagnostic for the SANA-Streaming server.

If the machine reboots while running server.py, run this instead. It exercises
each stage in isolation and prints a line (flushed) before each one, so the
last line on screen/journal tells you exactly which stage triggers the crash.

Usage:
    python diagnose_gpu.py            # all stages
    python diagnose_gpu.py stress     # pure torch compute stress only

Useful A/B runs when hunting a kernel-level hang (a whole-machine crash that
survives power/clock caps is usually a wedged CUDA/Triton kernel, not watts):

    CUDA_LAUNCH_BLOCKING=1 python diagnose_gpu.py
        Makes every kernel launch synchronous, so the last printed line really
        is the crashing stage (async launches can misattribute the crash to a
        later stage).

    DISABLE_FLASH_ATTN=1 python diagnose_gpu.py
        Routes the softmax attention layers through PyTorch SDPA instead of
        the flash-attn CUDA kernels. If the crash disappears, flash-attn
        (mis)built for your GPU arch is the culprit; run the server with
        DISABLE_FLASH_ATTN=1 too.

    TEXT_ENCODER_ATTN=eager python diagnose_gpu.py
        Forces the Gemma text encoder to plain PyTorch eager attention. Try
        this when the crash happens right after "[load] Weights loaded.
        Encoding prompt" / Stage 2b — no Triton kernels run that early, so
        the encoder's attention backend or cuBLAS is the suspect.

Watch power draw in another terminal while it runs:
    watch -n0.5 nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv

After any crash, check the kernel log from the previous boot:
    journalctl -k -b -1 | grep -i xid
Xid 13/31 = a kernel faulted (software); Xid 74 "GPU has fallen off the bus"
= hardware (connector/riser/PSU) despite power limits.
"""

import os
import sys
import time
from pathlib import Path

os.environ["DISABLE_XFORMERS"] = "1"
os.environ.setdefault("USE_CHUNKWISE_GDN", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

BACKEND_ROOT = Path(__file__).resolve().parent
SANA_ROOT = Path(os.environ.get("SANA_ROOT", BACKEND_ROOT / "Sana-main")).resolve()
if str(SANA_ROOT) not in sys.path:
    sys.path.insert(0, str(SANA_ROOT))


def stage(msg):
    print(f"\n=== {msg} ===", flush=True)
    torch.cuda.synchronize()


def stress_test():
    stage("Stage 0: pure bf16 matmul stress (no Sana code)")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for i in range(500):
        c = a @ b
        if i % 50 == 0:
            torch.cuda.synchronize()
            print(f"  iter {i} ok", flush=True)
    print("  stress test passed", flush=True)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
          f"device: {torch.cuda.get_device_name(0)} "
          f"(capability {torch.cuda.get_device_capability(0)})", flush=True)

    stress_test()
    if only == "stress":
        return

    import pyrallis
    from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae
    from diffusion.model.utils import get_weight_dtype
    from diffusion.utils.config import model_v2v_init_config
    from tools.download import find_model
    from server import (
        CONFIG_PATH,
        FRAME_HEIGHT,
        FRAME_WIDTH,
        STREAMING_REPO_ID,
        StreamingInferenceConfig,
        load_text_encoder,
    )

    config = pyrallis.parse(config_class=StreamingInferenceConfig, config_path=str(CONFIG_PATH), args=[])
    device = torch.device("cuda")
    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)

    stage("Stage 1: load VAE (Lightricks/LTX-2)")
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device=device, dtype=vae_dtype, config=config.vae)
    stage(f"Stage 1b: VAE encode one {FRAME_WIDTH}x{FRAME_HEIGHT} frame")
    from torchvision import transforms as T
    img = torch.rand(1, 3, 1, FRAME_HEIGHT, FRAME_WIDTH, device=device, dtype=vae_dtype) * 2 - 1
    out = vae.streaming_causal_encode(img, causal=True, reset_cache=True).latent_dist.mode()
    print(f"  latent shape: {tuple(out.shape)}", flush=True)

    stage("Stage 2: load text encoder (gemma-2-2b-it)")
    te_attn = os.environ.get("TEXT_ENCODER_ATTN")
    print(f"  text encoder attn_implementation: {te_attn or '(transformers default)'}", flush=True)
    if te_attn:
        tokenizer, text_encoder = load_text_encoder(config.text_encoder.text_encoder_name, device, te_attn)
    else:
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, device=device)
    stage("Stage 2b: gemma forward pass (first big compute)")
    tokens = tokenizer("hello world", max_length=300, padding="max_length",
                       truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        h = text_encoder(tokens.input_ids, tokens.attention_mask)[0]
    print(f"  embeds shape: {tuple(h.shape)}", flush=True)

    stage("Stage 3: build DiT (registers GDN/triton modules)")
    latent_size = config.model.image_size // config.vae.vae_downsample_rate
    model_kwargs = model_v2v_init_config(config, latent_size=latent_size)
    model = build_model(config.model.model,
                        use_fp32_attention=config.model.get("fp32_attention", False),
                        **model_kwargs).to(device)

    stage("Stage 4: download/load DiT checkpoint")
    state_dict = find_model(f"hf://{STREAMING_REPO_ID}/dit/sana_streaming_ar.pth")
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    if "state_dict" not in state_dict:
        state_dict = {"state_dict": {k.removeprefix("model.").removeprefix("module."): v
                                     for k, v in state_dict.items()}}
    state_dict = state_dict["state_dict"]
    if "pos_embed" not in state_dict and "pos_embed" in model.state_dict():
        state_dict["pos_embed"] = model.state_dict()["pos_embed"]
    model.load_state_dict(state_dict, strict=True)
    model = model.eval().to(weight_dtype)
    del state_dict
    torch.cuda.empty_cache()

    stage("Stage 5: one DiT forward (triton GDN kernels compile + run here)")
    from diffusion.utils.import_utils import is_flash_attn_available, is_xformers_available
    print(f"  attention backend: flash_attn={is_flash_attn_available()} "
          f"xformers={is_xformers_available()} "
          f"chunkwise_gdn={os.environ.get('USE_CHUNKWISE_GDN', '1')}", flush=True)
    cond = torch.randn(1, 300, h.shape[-1], device=device, dtype=weight_dtype)
    # The model raises "Attention type is not available" when mask=None and
    # xformers is disabled; the server always passes the prompt mask (B, L).
    mask = torch.ones(1, 300, device=device, dtype=torch.long)
    noise = torch.randn(1, config.vae.vae_latent_dim, 4, FRAME_HEIGHT // 32, FRAME_WIDTH // 32,
                        device=device, dtype=weight_dtype)
    src = torch.randn_like(noise).to(vae_dtype)
    t = torch.tensor([500], device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
        out, _ = model(noise, t, cond, start_f=0, end_f=4, frame_index=None,
                       save_kv_cache=True, kv_cache=[[None] * 6 for _ in model.blocks],
                       mask=mask, data_info={"image_vae_embeds": src})
    print(f"  output shape: {tuple(out.shape)}", flush=True)

    stage("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
