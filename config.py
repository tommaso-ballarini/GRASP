"""
Centralized configuration for the R2P-GEN pipeline.
All configurable parameters should be defined here.

Configuration is organized by module for clarity:
- Cluster: Cluster mode detection and base paths
- BuildDatabase: Settings for database/fingerprint extraction
- Database: Database file naming strategy
- Models: Model paths and identifiers
- GPU: GPU memory and performance settings
- Generate: Settings for image generation
- Refine: Refinement loop settings
- Images: Image sizing and processing
- Paths: Directory paths (cluster-aware)

Environment variables:
  R2P_CLUSTER_MODE=true        → abilita cluster mode
  R2P_MODELS_BASE=<path>       → override base directory modelli (cluster)
  HF_HOME=<path>               → override HuggingFace cache (gestito esternamente / SLURM)
"""
import os


# ---------------------------------------------------------------------------
# Helper: risolve path modelli in modo cluster-agnostico
# ---------------------------------------------------------------------------
_MODELS_BASE = os.environ.get("R2P_MODELS_BASE", "")

def _model_path(repo_id: str, local_dirname: str) -> str:
    """
    Restituisce il path locale se R2P_MODELS_BASE è impostato,
    altrimenti il repo-id HuggingFace (per download automatico).
    """
    if _MODELS_BASE:
        return os.path.join(_MODELS_BASE, local_dirname)
    return repo_id


# ---------------------------------------------------------------------------
# FIX NameError: le variabili cluster-aware vengono calcolate a livello di
# MODULO (fuori da Config), così sono disponibili a tutte le inner class
# senza dover referenziare Config.Cluster dall'interno di Config stessa.
# Python non permette alle inner class di vedere le classi-sorella definite
# prima nella stessa outer class durante la valutazione del corpo della classe.
# ---------------------------------------------------------------------------
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
        # HOME_DIR solo in cluster mode (retrocompatibilità)
        HOME_DIR = os.environ.get("HOME", "/home/user") if _CLUSTER_MODE else None

    # ========================================================================
    # BUILD_DATABASE CONFIGURATION
    # ========================================================================
    class BuildDatabase:
        """Configuration for pipeline/build_database.py"""
        # Usa le variabili di modulo — non Config.Cluster (causa NameError)
        SOURCE_DATA_DIR = os.path.join(_DATA_BASE, "perva-data")
        DATASET_SPLIT = "train"

        DEBUG_MODE = True
        DEBUG_LIMIT = 30

        USE_CLIP_CATEGORY = True
        USE_CLIP_SELECTION = True
        IGNORE_LAION = True
        SEED = 42

        # Options: 'simple', 'optimized', 'gemini'
        SDXL_PROMPT_STRATEGY = 'gemini'

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
        Tutti i modelli della pipeline.

        In cluster mode: imposta R2P_MODELS_BASE nel SLURM script o .bashrc.
        In locale: i repo_id vengono scaricati automaticamente da HuggingFace.
        """

        # --- verify / refine loop ---
        VLM_MODEL = _model_path(
            repo_id="openbmb/MiniCPM-o-2_6",
            local_dirname="MiniCPM-o-2_6",
        )

        # --- flux_loop.py reasoner (Qwen3-VL) ---
        QWEN3_MODEL = _model_path(
            repo_id="Qwen/Qwen3-VL-8B-Instruct",
            local_dirname="Qwen3-VL-8B-Instruct",
        )

        # --- Final Judge (InternVL3_5-8B) — diverso da verify/refine ---
        JUDGE_MODEL = _model_path(
            repo_id="OpenGVLab/InternVL3-8B",
            local_dirname="InternVL3_5-8B",
        )

        # --- Generatori ---
        SDXL_MODEL = _model_path(
            repo_id="stabilityai/stable-diffusion-xl-base-1.0",
            local_dirname="stable-diffusion-xl-base-1.0",
        )

        _FLUX_PATH = os.environ.get(
            "R2P_FLUX_MODEL",
            os.path.join(os.environ.get("R2P_MODELS_BASE", ""), "FLUX.2-klein-9B")
                if os.environ.get("R2P_MODELS_BASE") else "black-forest-labs/FLUX.1-schnell"
        )
        FLUX_MODEL = _FLUX_PATH

        # --- IP-Adapter ---
        IP_ADAPTER_REPO = "h94/IP-Adapter"
        IP_ADAPTER_SUBFOLDER = "sdxl_models"
        IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sdxl.bin"

        CLIP_MODEL = os.path.join(
            "/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface",
            "clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
        )

        CLIP_MODEL_336 = os.path.join(
            "/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface",
            "clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
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
        IP_ADAPTER_SCALE_GLOBAL = 0.6
        NUM_INFERENCE_STEPS = 40
        GUIDANCE_SCALE = 7.5
        SEED = 42

        SDXL_USE_PROMPT_WEIGHTS = True
        SDXL_SUBJECT_WEIGHT = 1.2

        # Options: "white", "wooden_table", "gradient", "neutral", "none"
        SDXL_BACKGROUND_STYLE = "wooden_table"

        SDXL_BACKGROUND_TEMPLATES = {
            "white":        "placed on clean white surface, seamless white background",
            "wooden_table": "placed on wooden table, soft shadows, neutral background",
            "gradient":     "isolated on gradient white background, professional product shot",
            "neutral":      "on neutral surface, studio background",
            "none":         "",
        }

        SDXL_QUALITY_SUFFIX = "professional product photography, 8k, sharp focus"

        NEGATIVE_PROMPT = (
            "blurry, low quality, low resolution, distorted, deformed, "
            "(background contamination:1.3), (reference background leakage:1.2), "
            "(original background visible:1.2), "
            "artifact, watermark, text overlay, logo overlay, signature, "
            "oversaturated, overexposed, underexposed, noise, grain, "
            "worst quality, jpeg artifacts, duplicate, cropped, "
            "unrealistic proportions, anatomical errors, "
            "blur, out of focus"
        )

        USE_LAYERWISE_SCALING = True
        IP_ADAPTER_LAYER_WEIGHTS = {
            "down": {
                "block_0": [0.0, 0.0],
                "block_1": [0.0, 0.0],
                "block_2": [0.4, 0.7],
            },
            "mid": 0.9,
            "up": {
                "block_0": [0.6, 0.8, 0.9],
                "block_1": [0.95, 0.95, 0.95],
                "block_2": [0.85, 0.85],
                "block_3": [0.7],
            }
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
        # Usa variabili di modulo — stesso motivo del fix BuildDatabase
        OUTPUT_DIR = os.environ.get("R2P_OUTPUT_DIR", os.path.join(_BASE_DIR, "output")) if _CLUSTER_MODE else "output"
        DATASET_DIR  = os.path.join(_DATA_BASE, "perva-data") if _CLUSTER_MODE else "data/perva-data"
        DATABASE_DIR = os.path.join(_BASE_DIR, "database") if _CLUSTER_MODE else "database"

    # ========================================================================
    # BACKWARD COMPATIBILITY ALIASES
    # ========================================================================
    DEVICE          = GPU.DEVICE
    USE_FP16        = GPU.USE_FP16
    VLM_MODEL       = Models.VLM_MODEL
    TARGET_ACCURACY = Refine.TARGET_ACCURACY
    MAX_IMAGE_DIM   = Images.MAX_IMAGE_DIM

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    @classmethod
    def get_background_template(cls) -> str:
        """Get the current background template string."""
        return cls.Generate.SDXL_BACKGROUND_TEMPLATES.get(
            cls.Generate.SDXL_BACKGROUND_STYLE,
            cls.Generate.SDXL_BACKGROUND_TEMPLATES["white"]
        )

    @classmethod
    def print_summary(cls) -> None:
        """Stampa un riepilogo della configurazione attiva."""
        print("=" * 60)
        print("R2P-GEN Config Summary")
        print("=" * 60)
        print(f"  Cluster mode : {cls.Cluster.MODE}")
        print(f"  R2P_MODELS_BASE: {_MODELS_BASE or '(not set — HuggingFace Hub)'}")
        print(f"  VLM (verify) : {cls.Models.VLM_MODEL}")
        print(f"  Qwen3 (flux) : {cls.Models.QWEN3_MODEL}")
        print(f"  Judge        : {cls.Models.JUDGE_MODEL}")
        print(f"  FLUX         : {cls.Models.FLUX_MODEL}")
        print(f"  Device       : {cls.GPU.DEVICE}  |  FP16: {cls.GPU.USE_FP16}")
        print(f"  Max iter     : {cls.Refine.MAX_ITERATIONS}")
        print(f"  Target acc   : {cls.Refine.TARGET_ACCURACY:.0%}")
        print("=" * 60)