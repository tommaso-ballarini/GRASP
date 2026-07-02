#!/usr/bin/env python3
"""
Download all required models to HuggingFace cache.
Run this BEFORE submitting pipeline jobs.
"""
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer, AutoProcessor

def main():
    print("=" * 60)
    print("R2P-GEN Model Downloader")
    print("=" * 60)
    
    models = [
        ("openbmb/MiniCPM-o-2_6", "MiniCPM (VLM) - ~15GB"),
        ("stabilityai/stable-diffusion-xl-base-1.0", "SDXL Base - ~7GB"),
        ("h94/IP-Adapter", "IP-Adapter - ~100MB"),
        ("Qwen/Qwen2-VL-7B-Instruct", "Qwen2-VL (Judge) - ~15GB"),
        ("openai/clip-vit-large-patch14-336", "CLIP (336) - ~1GB"),
        ("openai/clip-vit-large-patch14", "CLIP (224) - ~1GB"),
    ]
    
    for model_id, desc in models:
        print(f"\n{'─' * 60}")
        print(f"📥 Downloading: {desc}")
        print(f"   Model ID: {model_id}")
        print(f"{'─' * 60}")
        try:
            snapshot_download(repo_id=model_id)
            print(f"   ✅ Success!")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # DINO (from torch hub)
    print(f"\n{'─' * 60}")
    print("📥 Downloading: DINO v2 (from torch hub)")
    print(f"{'─' * 60}")
    try:
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        print("   ✅ Success!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Model download complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()