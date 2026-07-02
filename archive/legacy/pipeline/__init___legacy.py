# pipeline/__init__.py
"""
R2P-GEN Pipeline Module

This module contains the core components of the R2P-GEN pipeline:
- build_database: Database building and fingerprint extraction
- generate: Image generation using SDXL + IP-Adapter
- verify: Generated image verification (branch: verify)
- refine: Iterative refinement loop with feedback
- judge: Final evaluation with VQA + metrics (CLIP, DINO, TIFA)
- metrics: Quantitative evaluation metrics
- utils2: Utility functions for the pipeline
"""

from .build_database import DatabaseBuilder
from .generate import Generator
from .refine import iterative_refinement, build_negative_from_failed
from .judge import FinalJudge, JudgeResult
from .metrics import MetricsCalculator, MetricsResult, quick_evaluate
from .utils2 import cleanup_gpu, print_memory_stats, ensure_output_dir, get_iteration_filename

__all__ = [
    # Core pipeline
    'DatabaseBuilder',
    'Generator',
    'iterative_refinement',
    
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
    'build_negative_from_failed',
]
