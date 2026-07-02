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
"""
import os

class Config:
    # ========================================================================
    # CLUSTER MODE CONFIGURATION
    # ========================================================================
    class Cluster:
        """Cluster mode detection and base paths"""

        # Set to True when running on SLURM cluster
        # Can be controlled via environment variable: R2P_CLUSTER_MODE=true
        MODE = os.environ.get("R2P_CLUSTER_MODE", "false").lower() == "true"

        # Base paths (auto-detect based on mode)
        if MODE:
            HOME_DIR = os.environ.get("HOME", "/home/tommaso.ballarini-1")
            BASE_DIR = os.path.join(HOME_DIR, "R2P-GEN")
            DATA_BASE = os.path.join(HOME_DIR, "data")
        else:
            BASE_DIR = "."
            DATA_BASE = "data"

    # ========================================================================
    # BUILD_DATABASE CONFIGURATION
    # ========================================================================
    class BuildDatabase:
        """Configuration for pipeline/build_database.py"""

        # Dataset Configuration
        SOURCE_DATA_DIR = os.path.join(Cluster.DATA_BASE, "perva-data")  # Cluster-aware
        DATASET_SPLIT = "train"              # Options: "train", "test", "all"

        # Run Settings
        DEBUG_MODE = True                    # Set to False to process the entire dataset
        DEBUG_LIMIT = 5                      # Number of concepts to process in debug mode

        # Feature Extraction Settings
        USE_CLIP_CATEGORY = True             # True = Auto-detect category via CLIP; False = Use folder name
        USE_CLIP_SELECTION = True            # True = Select most representative image via CLIP centroid
                                              # False = Use first image (sorted numerically)
        IGNORE_LAION = True                  # Ignore 'laion' subdirectories (R2P training data)
        SEED = 42                            # Random seed for reproducible CLIP selection

        # SDXL Prompt Generation Strategy
        # Options: 'simple', 'optimized', 'gemini'
        SDXL_PROMPT_STRATEGY = 'gemini'      # 'simple' = Natural description + style suffix (baseline)
                                              # 'optimized' = Hierarchical tags with weights (R2P enhanced)
                                              # 'gemini' = Brand-first, ultra-concise (SOTA personalization)

    # ========================================================================
    # DATABASE NAMING CONFIGURATION
    # ========================================================================
    class Database:
        """Database file naming strategy"""

        # True  → "database.json"  (canonical, for main branch / production)
        # False → "database_perva_{split}_{method}.json" (descriptive, for testing branches)
        CANONICAL_NAME = True

    # ========================================================================
    # MODELS CONFIGURATION
    # ========================================================================
    class Models:
        """Model paths and identifiers"""

        VLM_MODEL = "openbmb/MiniCPM-o-2_6"
        SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
        IP_ADAPTER_REPO = "h94/IP-Adapter"
        IP_ADAPTER_SUBFOLDER = "sdxl_models"
        IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sdxl.bin"
        QWEN_MODEL = "/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface/Qwen3-VL-8B-Instruct"  # For Final Judge (judge.py)
        FLUX_MODEL = "/leonardo_work/IscrC_MUSE/tballari/models_cache/FLUX.2-klein-9B"

    # ========================================================================
    # GPU CONFIGURATION
    # ========================================================================
    class GPU:
        """GPU memory and performance settings"""

        DEVICE = "cuda"
        MEMORY_FRACTION = 0.9
        USE_FP16 = True
        # Flush VRAM cache every N generations in the batch Generator loop.
        # Set to 0 to disable mid-loop cache clearing (only cleared at cleanup()).
        CLEAR_CACHE_EVERY = 10

    # ========================================================================
    # GENERATION CONFIGURATION
    # ========================================================================
    class Generate:
        """Configuration for pipeline/generate.py"""

        IP_ADAPTER_SCALE_GLOBAL = 0.6
        NUM_INFERENCE_STEPS = 40
        GUIDANCE_SCALE = 7.5
        SEED = 42  # Base seed: passed as torch.Generator to every SDXL .pipe() call

        # Use weights in SDXL prompts (e.g., "(bag:1.2)")
        # Set to False for cleaner prompts with more refinement flexibility
        SDXL_USE_PROMPT_WEIGHTS = True

        # Light weight for main subject (only used if SDXL_USE_PROMPT_WEIGHTS=True)
        SDXL_SUBJECT_WEIGHT = 1.2

        # Background specification for generated images
        # Options: "white", "wooden_table", "gradient", "neutral", "none"
        SDXL_BACKGROUND_STYLE = "wooden_table"

        # Background templates (used in prompt generation)
        SDXL_BACKGROUND_TEMPLATES = {
            "white": "placed on clean white surface, seamless white background",
            "wooden_table": "placed on wooden table, soft shadows, neutral background",
            "gradient": "isolated on gradient white background, professional product shot",
            "neutral": "on neutral surface, studio background",
            "none": ""  # No background specification (not recommended)
        }

        # Standard quality suffix for all prompts
        SDXL_QUALITY_SUFFIX = "professional product photography, 8k, sharp focus"

        # Negative prompt to prevent artifacts
        # Prevents background contamination and common SDXL artifacts
        # Critical for layer-wise scaling: reinforces down blocks=0.0
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

        # IP-ADAPTER LAYER-WISE SCALING
        # Strategy: Minimize background contamination, maximize identity preservation
        # Based on InstantStyle research + R2P optimization
        USE_LAYERWISE_SCALING = True
        IP_ADAPTER_LAYER_WEIGHTS = {
            "down": {
                "block_0": [0.0, 0.0],         # Zero background from reference
                "block_1": [0.0, 0.0],         # Zero composition from reference
                "block_2": [0.4, 0.7],         # Object shape preservation (smooth gradient)
            },
            "mid": 0.9,                        # Very high semantic identity
            "up": {
                "block_0": [0.6, 0.8, 0.9],    # Texture/color injection (3 layers)
                "block_1": [0.95, 0.95, 0.95], # Maximum material fidelity (3 layers)
                "block_2": [0.85, 0.85],       # Fine details preservation (2 layers)
                "block_3": [0.7],              # Final refinement (1 layer)
            }
        }

    # ========================================================================
    # REFINEMENT LOOP CONFIGURATION
    # ========================================================================
    class Refine:
        """Configuration for pipeline/refine.py"""

        MAX_ITERATIONS = 3
        TARGET_ACCURACY = 0.95  # 95% accuracy to exit loop
        MIN_IMPROVEMENT = 0.05  # Minimum improvement between iterations

    # ========================================================================
    # IMAGES CONFIGURATION
    # ========================================================================
    class Images:
        """Image sizing and processing settings"""

        MAX_IMAGE_DIM = 896
        FALLBACK_DIM = 448      # Reduced dimension in case of OOM
        REFERENCE_IMAGE_SIZE = 512  # Size to resize reference images for IP-Adapter
        OUTPUT_IMAGE_SIZE = 1024    # Generated image size

    # ========================================================================
    # PATHS CONFIGURATION (Cluster-Aware)
    # ========================================================================
    class Paths:
        """Directory paths (auto-adjusted for cluster mode)"""

        OUTPUT_DIR = os.path.join(Cluster.BASE_DIR, "output") if Cluster.MODE else "output"
        DATASET_DIR = os.path.join(Cluster.DATA_BASE, "perva-data") if Cluster.MODE else "data/perva-data"
        DATABASE_DIR = os.path.join(Cluster.BASE_DIR, "database") if Cluster.MODE else "database"

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