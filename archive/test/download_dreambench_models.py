"""
download_dreambench_models.py

Da lanciare SOLO sul login node (ha accesso a internet).
Scarica e cachea localmente i due checkpoint usati dalla Fase 4
(metriche ufficiali DreamBench), così che i job sui nodi di calcolo
(offline) possano caricarli da disco con HF_HUB_OFFLINE=1.

Uso:
    python download_dreambench_models.py
"""

import os

# Stesso HF_HOME usato negli sbatch, per coerenza di cache.
os.environ.setdefault(
    "HF_HOME",
    "/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface",
)

from transformers import CLIPModel, CLIPProcessor, ViTModel, ViTImageProcessor

CLIP_ID = "openai/clip-vit-base-patch32"   # standard DreamBooth/DreamBench
DINO_ID = "facebook/dino-vits16"           

# download_dreambench_models.py — blocco corretto

print(f"📦 Scarico CLIP: {CLIP_ID}")
CLIPModel.from_pretrained(CLIP_ID, use_safetensors=True)  # ha safetensors, ok
CLIPProcessor.from_pretrained(CLIP_ID)

# DINO: i pesi vengono scaricati e convertiti da convert_dino_to_safetensors.py
# Qui scarica solo config e processor
print(f"📦 Scarico config/processor DINO: {DINO_ID}")
ViTImageProcessor.from_pretrained(DINO_ID)
print("   ⚠️  Per i pesi DINO lancia: python convert_dino_to_safetensors.py")

print("\n✅ Download completato.")
print(f"   HF_HOME usato: {os.environ['HF_HOME']}")
print("   Verifica che config.py punti agli snapshot corretti, es:")
print(f"     find {os.environ['HF_HOME']} -iname '*clip-vit-base-patch32*' -maxdepth 3")
print(f"     find {os.environ['HF_HOME']} -iname '*dino-vits16*' -maxdepth 3")