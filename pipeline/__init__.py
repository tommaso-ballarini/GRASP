"""
R2P-GEN Pipeline Module (FLUX Edition)

Components orchestrated by flux_loop.py:
  - generate: Image generation with FLUX Img2Img
  - verify: Verification with Qwen3-VL + CLIP
  - judge: Final evaluation with InternVL3_5-8B + CLIP/DINO/VQA
  - metrics: Quantitative metrics (CLIP-I, CLIP-T, DINO-I, TIFA)
  - utils2: GPU memory management and file handling utilities
"""

from .generate import Generator
from .verify import verify_generation_r2p
from .judge import FinalJudge, JudgeResult
from .metrics import MetricsCalculator, MetricsResult, quick_evaluate
from .utils2 import cleanup_gpu, print_memory_stats, ensure_output_dir, get_iteration_filename

__all__ = [
    # Core pipeline (FLUX)
    'Generator',

    # Verification
    'verify_generation_r2p',

    # Final Judge & Metrics
    'FinalJudge',
    'JudgeResult',
    'MetricsCalculator',
    'MetricsResult',
    'quick_evaluate',

    # Utilities
    'cleanup_gpu',
    'print_memory_stats',
    'ensure_output_dir',
    'get_iteration_filename',
]