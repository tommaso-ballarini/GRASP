"""
Centralized configuration for the R2P-GEN pipeline (FLUX Edition).

Environment variables:
  R2P_CLUSTER_MODE=true       → to allow cluster mode
  R2P_MODELS_BASE=<path>      → override base directory of models (cluster)
  HF_HOME=<path>              → override HuggingFace cache 
  R2P_FLUX_MODEL=<path>       → override path modello FLUX
"""
import os


# ---------------------------------------------------------------------------
# Helper function to determine model paths based on environment variables.
# ---------------------------------------------------------------------------
_MODELS_BASE = os.environ.get("R2P_MODELS_BASE", "")

def _model_path(repo_id: str, local_dirname: str) -> str:
    """
    Returns the local path if R2P_MODELS_BASE is set,
    otherwise the HuggingFace repo-id (for automatic download).
    """
    if _MODELS_BASE:
        return os.path.join(_MODELS_BASE, local_dirname)
    return repo_id

_CLUSTER_MODE = os.environ.get("R2P_CLUSTER_MODE", "false").lower() == "true"

if _CLUSTER_MODE:
    _HOME_DIR = os.environ.get("HOME", "/home/user")
    _BASE_DIR = os.path.join(_HOME_DIR, "R2P-GEN")
    _DATA_BASE = os.path.join(_HOME_DIR, "data")
else:
    _BASE_DIR = "."
    _DATA_BASE = "data"


class Config:
    # ========================================================================
    # CLUSTER MODE CONFIGURATION
    # ========================================================================
    class Cluster:
        """Cluster mode detection and base paths."""
        MODE     = _CLUSTER_MODE
        BASE_DIR = _BASE_DIR
        DATA_BASE = _DATA_BASE
        HOME_DIR = os.environ.get("HOME", "/home/user") if _CLUSTER_MODE else None

    # ========================================================================
    # BUILD_DATABASE CONFIGURATION
    # ========================================================================
    class BuildDatabase:
        """Configuration for pipeline/build_database.py"""
        SOURCE_DATA_DIR = os.path.join(_DATA_BASE, "perva-data")
        DATASET_SPLIT = "train"

        DEBUG_MODE = False
        DEBUG_LIMIT = 30

        USE_CLIP_CATEGORY = True
        USE_CLIP_SELECTION = True
        IGNORE_LAION = True
        SEED = 42

    # ========================================================================
    # DATABASE NAMING CONFIGURATION
    # ========================================================================
    class Database:
        """Database file naming strategy."""
        # True  → "database.json"  (produzione)
        # False → "database_perva_{split}_{method}.json"  (test/dev)
        CANONICAL_NAME = True

    # ========================================================================
    # MODELS CONFIGURATION
    # ========================================================================
    class Models:
        """
        All models used in the R2P-GEN pipeline (reasoner, judge, FLUX, CLIP, DINO).

        In cluster mode: set R2P_MODELS_BASE in SLURM script o .bashrc.
        In local mode: repo_id are automatically downloaded from HuggingFace.
        """

        # --- flux_loop.py reasoner (Qwen3-VL) ---
        QWEN3_MODEL = _model_path(
            repo_id="Qwen/Qwen3-VL-8B-Instruct",
            local_dirname="Qwen3-VL-8B-Instruct",
        )

        # --- Final Judge (InternVL3_5-8B) ---
        JUDGE_MODEL = _model_path(
            repo_id="OpenGVLab/InternVL3-8B",
            local_dirname="InternVL3_5-8B",
        )

        _FLUX_PATH = os.environ.get(
            "R2P_FLUX_MODEL",
            os.path.join(os.environ.get("R2P_MODELS_BASE", ""), "FLUX.2-klein-9B")
                if os.environ.get("R2P_MODELS_BASE") else "black-forest-labs/FLUX.1-schnell"
        )
        FLUX_MODEL = _FLUX_PATH
        FLUX_TEXT_URL = "http://127.0.0.1:8767"

        CLIP_MODEL = os.path.join(
            "/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface",
            "clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
        )

        CLIP_MODEL_336 = os.path.join(
            "/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface",
            "clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
        )

        DINO_MODEL = "/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface/dinov2-large"

        # === Official DreamBench metrics ===
        # different from the models used in original pipeline, but they are the ones used in the DreamBench paper
        CLIP_DREAMBENCH_MODEL = os.path.join(
            "/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface",
            "hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
        )
        
        DINO_DREAMBENCH_MODEL = os.path.join(
            "/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface",
            "hub/models--facebook--dino-vits16/snapshots/abe3b354cb6a9b6f146096b14a4a9d7eecbcb4bd"
        )
    
    # ========================================================================
    # GPU CONFIGURATION
    # ========================================================================
    class GPU:
        """GPU memory and performance settings."""
        DEVICE = "cuda"
        MEMORY_FRACTION = 0.9
        USE_FP16 = True
        CLEAR_CACHE_EVERY = 10

    # ========================================================================
    # GENERATION CONFIGURATION
    # ========================================================================
    class Generate:
        """Configuration for pipeline/generate.py"""
        NUM_INFERENCE_STEPS = 4
        GUIDANCE_SCALE = 7.5 #NOT IN USE WITH FLUX.2 --> GUIDANCE SCALE IS SET TO 0.0
        SEED = 42

        # Options: "white", "wooden_table", "gradient", "neutral", "none"
        BACKGROUND_STYLE = "wooden_table"

        BACKGROUND_TEMPLATES = {
            "white":        "placed on clean white surface, seamless white background",
            "wooden_table": "placed on wooden table, soft shadows, neutral background",
            "gradient":     "isolated on gradient white background, professional product shot",
            "neutral":      "on neutral surface, studio background",
            "none":         "",
        }

    # ========================================================================
    # REFINEMENT LOOP CONFIGURATION
    # ========================================================================
    class Refine:
        """Configuration for pipeline/refine.py and flux_loop.py"""
        MAX_ITERATIONS = 3
        TARGET_ACCURACY = 0.95
        MIN_IMPROVEMENT = 0.05

    # ========================================================================
    # IMAGES CONFIGURATION
    # ========================================================================
    class Images:
        """Image sizing and processing settings."""
        MAX_IMAGE_DIM = 896
        FALLBACK_DIM = 448
        REFERENCE_IMAGE_SIZE = 512
        OUTPUT_IMAGE_SIZE = 1024

    # ========================================================================
    # PATHS CONFIGURATION (Cluster-Aware)
    # ========================================================================
    class Paths:
        """Directory paths (auto-adjusted for cluster mode)."""

        OUTPUT_DIR = os.environ.get("R2P_OUTPUT_DIR", os.path.join(_BASE_DIR, "output")) if _CLUSTER_MODE else "output"
        DATASET_DIR  = os.path.join(_DATA_BASE, "perva-data") if _CLUSTER_MODE else "data/perva-data"
        DATABASE_DIR = os.path.join(_BASE_DIR, "database") if _CLUSTER_MODE else "database"

    # ========================================================================
    # BACKWARD COMPATIBILITY ALIASES
    # ========================================================================
    
    DEVICE          = GPU.DEVICE
    USE_FP16        = GPU.USE_FP16
    TARGET_ACCURACY = Refine.TARGET_ACCURACY
    MAX_IMAGE_DIM   = Images.MAX_IMAGE_DIM

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    @classmethod
    def get_background_template(cls) -> str:
        """Get the current background template string."""
        return cls.Generate.BACKGROUND_TEMPLATES.get(
            cls.Generate.BACKGROUND_STYLE,
            cls.Generate.BACKGROUND_TEMPLATES["white"]
        )

    @classmethod
    def print_summary(cls) -> None:
        """Log the current configuration settings to the console."""
        print("=" * 60)
        print("R2P-GEN Config Summary")
        print("=" * 60)
        print(f"  Cluster mode : {cls.Cluster.MODE}")
        print(f"  R2P_MODELS_BASE: {_MODELS_BASE or '(not set — HuggingFace Hub)'}")
        print(f"  Reasoner     : {cls.Models.QWEN3_MODEL}")
        print(f"  Judge        : {cls.Models.JUDGE_MODEL}")
        print(f"  FLUX         : {cls.Models.FLUX_MODEL}")
        print(f"  Device       : {cls.GPU.DEVICE}  |  FP16: {cls.GPU.USE_FP16}")
        print(f"  Max iter     : {cls.Refine.MAX_ITERATIONS}")
        print(f"  Target acc   : {cls.Refine.TARGET_ACCURACY:.0%}")
        print("=" * 60)