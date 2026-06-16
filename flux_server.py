"""
flux_server.py
==============
Server FastAPI minimale che espone FLUX come endpoint HTTP.
Lanciato da recovery_pipeline.sh in background con flux_test_work.
Gira sulla porta 8766.
"""

import os
import io
import base64
import torch
import argparse  
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ── Patch Diffusers (identica a quella in recovery_pipeline_v2.py) ──────────
import diffusers.models.attention_processor
import torch.nn.functional as F

_original_sdpa = F.scaled_dot_product_attention
def _stripped_sdpa(*args, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _original_sdpa(*args, **kwargs)
diffusers.models.attention_processor.F.scaled_dot_product_attention = _stripped_sdpa

from diffusers import DiffusionPipeline

# ── Configurazione ───────────────────────────────────────────────────────────
FLUX_MODEL_DIR = "/leonardo_work/IscrC_MUSE/tballari/models_cache/FLUX.2-klein-9B"
STEPS          = 4


# ── Caricamento modello (una volta sola all'avvio) ───────────────────────────
print("🚀 Caricamento FLUX in VRAM...")
pipe = DiffusionPipeline.from_pretrained(
    FLUX_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
).to("cuda:0")
print("✅ FLUX pronto.")

app = FastAPI()


# ── Schemi request/response ──────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompts: list[str]
    seeds: list[int]
    source_image_b64: Optional[list[str]] = None  # lista di immagini base64, o None per text2img


class GenerateResponse(BaseModel):
    images_b64: list[str]   # immagini risultato in base64
    errors: list[str]       # stringa vuota se OK, messaggio di errore altrimenti


# ── Endpoint ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    images_b64 = []
    errors = []

    has_img = req.source_image_b64 is not None

    try:
        generators = [
            torch.Generator(device="cuda:0").manual_seed(s)
            for s in req.seeds
        ]

        kwargs = {
            "prompt": req.prompts,
            "num_inference_steps": STEPS,
            "generator": generators,
        }

        if has_img:
            source_images = []
            for b64 in req.source_image_b64:
                img_bytes = base64.b64decode(b64)
                source_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            kwargs["image"] = source_images

        outputs = pipe(**kwargs).images

        for img in outputs:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            images_b64.append(base64.b64encode(buf.getvalue()).decode())
            errors.append("")

    except Exception as e:
        # Se FLUX crasha sull'intero batch, segna errore per ogni immagine
        for _ in req.prompts:
            images_b64.append("")
            errors.append(f"FLUX Crash: {e}")

    return GenerateResponse(images_b64=images_b64, errors=errors)


# ── Avvio ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server FLUX via FastAPI")
    parser.add_argument("--port", type=int, default=8766, help="Porta su cui avviare il server")
    args = parser.parse_args()
    
    print(f"🌐 Avvio Uvicorn sulla porta dinamica: {args.port}...")
    
    uvicorn.run(app, host="127.0.0.1", port=args.port)