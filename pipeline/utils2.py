# utils2.py
"""
Utility functions for the R2P-GEN pipeline.
Provides GPU memory management and file handling utilities.
"""
import torch
import gc
import os
from pathlib import Path

def cleanup_gpu():
    """Aggressively free GPU memory."""
    gc.collect()                          # prima: distruggi oggetti Python senza riferimenti
    if torch.cuda.is_available():
        torch.cuda.empty_cache()          # poi: restituisci la memoria liberata al sistema
        torch.cuda.synchronize()          # infine: attendi che CUDA abbia completato

def print_memory_stats(label=""):
    """Print GPU memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        print(f"   📊 {label}")
        print(f"      Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Free: {free:.2f}GB")

def ensure_output_dir(output_dir="output"):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir

def get_iteration_filename(base_name, iteration):
    """Generate filename for a specific iteration."""
    stem = Path(base_name).stem
    ext = Path(base_name).suffix
    return f"{stem}_iter{iteration}{ext}"