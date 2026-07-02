# convert_dino_to_safetensors.py
# Lancia sul login node PRIMA del job di valutazione

import torch
from safetensors.torch import save_file
import os

SNAPSHOT = (
    "/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface"
    "/hub/models--facebook--dino-vits16/snapshots"
    "/abe3b354cb6a9b6f146096b14a4a9d7eecbcb4bd"
)
BIN_PATH  = os.path.join(SNAPSHOT, "pytorch_model.bin")
SAFE_PATH = os.path.join(SNAPSHOT, "model.safetensors")

# Step 1: scarica i pesi se non ci sono ancora
if not os.path.exists(BIN_PATH):
    url = (
        "https://huggingface.co/facebook/dino-vits16/resolve/main/pytorch_model.bin"
    )
    print(f"📥 Scarico {url}")
    os.system(f'wget -q --show-progress -O "{BIN_PATH}" "{url}"')
else:
    print(f"✅ pytorch_model.bin già presente")

# Step 2: converti in safetensors
# torch.load con weights_only=False è intenzionale qui:
# siamo sul login node, il file viene da facebook/HuggingFace, è trusted.
print("🔄 Conversione in safetensors...")
state_dict = torch.load(BIN_PATH, map_location="cpu", weights_only=False)
save_file(state_dict, SAFE_PATH)
print(f"✅ Salvato: {SAFE_PATH}")

# Step 3: verifica
from safetensors import safe_open
with safe_open(SAFE_PATH, framework="pt") as f:
    keys = list(f.keys())
print(f"   Chiavi nel file: {len(keys)} (attese ~202 per ViT-S/16)")