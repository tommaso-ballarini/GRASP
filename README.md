# R2P-GEN: Retrieval and Reasoning for Personalization — Generative Edition

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="FLUX.2" src="https://img.shields.io/badge/Generator-FLUX.2--klein--9B-orange">
  <img alt="Qwen3-VL" src="https://img.shields.io/badge/Reasoner-Qwen3--VL--8B-green">
  <img alt="InternVL3.5" src="https://img.shields.io/badge/Judge-InternVL3.5-yellow">
</p>

---

## Abstract

**R2P-GEN** is a zero-shot, tuning-free framework for subject-driven Text-to-Image personalization. Existing adapter-based methods (e.g., IP-Adapter) suffer from background entanglement and prompt override, while parameter-update approaches (e.g., DreamBooth, LoRA) impose prohibitive computational costs at inference time. Our method addresses both limitations by coupling the native generative power of **FLUX.2**, leveraging its T5 text encoder for dense semantic binding, with the visual reasoning capabilities of **Vision Language Models (VLMs)**. Subject identity is encoded entirely through structured textual fingerprints, eliminating any need for adapter modules or weight fine-tuning. A closed-loop, VLM-guided refinement stage automatically recovers missing visual attributes through escalating prompt revision, and an isolated independent judge provides unbiased final evaluation.

---

## Key Features

- **Zero-Shot & Tuning-Free:** No adapter weights, no LoRA, no per-subject fine-tuning. The framework runs entirely at inference time.
- **T5-Driven Dense Textual Anchoring:** Visual identity is encoded as structured natural language, exploiting the superior semantic binding capacity of the T5 encoder in FLUX.2.
- **VLM Visual Fingerprinting:** Qwen3-VL extracts distinctive attributes (logos, textures, typographic elements) from reference images into a structured JSON database.
- **Hierarchical Attribute Verification:** A two-phase verification module detects missing attributes with continuous probability scores.
- **Closed-Loop Refinement:** An automated recovery loop rewrites the generation prompt with increasing intensity (semantic emphasis → hard typographic anchors) and regenerates up to three times.
- **Independent Final Judge:** A fully isolated VLM (InternVL3.5) evaluates final outputs with Visual Question Answering (VQA), preventing self-evaluation bias.

---

## Pipeline Architecture

R2P-GEN is a five-stage modular pipeline orchestrated by `flux_loop.py`.

<p align="center">
  <img src="diagram_architecture.png" alt="R2P-GEN Architecture Diagram" width="100%">
</p>

### Stage Details

**1. FINGERPRINTS EXTRACTION**
* **Stage 0. Build Database (`pipeline/build_database.py`):** Processes a subject dataset (PerVA or DreamBench). A text-driven semantic view selection step via CLIP discards noisy views to find the reference image. Qwen3-VL then extracts structured visual fingerprints (logos, textures, typography) and serializes them into `database/database.json`.

**2. GENERATION**
* **Stage 1. Generation (`pipeline/generate.py`):** The JSON fingerprints are translated into a dense Textual Anchor. This anchor and the reference image are passed to FLUX.2-klein-9B in Image-to-Image mode (4 inference steps) to generate the personalized subject without any external adapters.

**3. REFINE LOOP AND JUDGE**
* **Stage 2. Verify Base (`pipeline/verify.py`):** Hierarchical attribute verification. A CLIP quick-reject filters catastrophic failures, followed by a Qwen3-VL logit-based sweep to compute a continuous probability for each attribute. A *Worst-K Detection* policy flags missing features.
* **Stage 3. Refine (`pipeline/refine.py`):** Rejected images enter a closed recovery loop. Qwen3-VL acts as an automated Prompt Engineer, applying an *Escalating Prompt Revision* strategy (semantic emphasis → hard typographic anchors) and regenerates the image up to 3 times.
* **Stage 4. Final Judge (`pipeline/judge`, `pipeline/metrics`):** To eliminate self-evaluation bias, InternVL3.5-8B evaluates the final outputs using CLIP-I (global identity similarity), CLIP-T (text-image alignment), DINO-I via DINOv2 (fine-grained structural fidelity), and a VQA-based TIFA Score (exact attribute verification).

---

## Repository Structure

```text
R2P-GEN/
├── flux_loop.py               # Main orchestrator for Stages 1-4
├── flux_server.py             # HTTP server to keep FLUX.2 resident in VRAM
├── config.py                  # Centralized configuration and model paths
├── pipeline/                  # Core pipeline modules
│   ├── build_database.py      # Stage 0 logic (fingerprinting)
│   ├── generate.py            # Stage 1 logic (FLUX inference)
│   ├── verify.py              # Stage 2 logic (CLIP/Qwen verification)
│   ├── refine.py              # Stage 3 logic (Prompt rewriting)
│   ├── judge.py               # Stage 4 logic (InternVL metrics)
│   ├── prompts/               # Prompt templates (flux_prompts.py, etc.)
│   ├── metrics.py             # Metric computation utilities
│   └── r2p_tools.py           # Shared utilities
├── r2p_core/                  # Underlying model wrappers
│   ├── evaluators/            # Confidence calculation logic
│   ├── models/                # Qwen3-VL and InternVL interface logic
│   └── utils/                 # Helpers
├── scripts/                   # Slurm cluster scripts for large-scale runs
│   ├── run_full_e2e.sh        # End-to-end pipeline run (All Stages)
│   ├── run_dreambench.sh      # DreamBench specific execution
│   └── run_ablation_*.sh      # Scripts for ablation studies
├── database/                  # Output directory for JSON databases
├── output/                    # Output directory for generated images
├── requeriments_cluster.txt   # Python depencecies for cluster dependencies
└── requirements.txt           # Python dependencies

```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (≥ 24 GB VRAM recommended; 40+ GB for full pipeline)
- [Conda](https://docs.conda.io/en/latest/) or a Python virtual environment

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/tommaso-ballarini/R2P-GEN.git
cd R2P-GEN

# Create and activate a conda environment
conda create -n r2pgen python=3.10 -y
conda activate r2pgen

# Install dependencies
pip install -r requirements.txt
```

For cluster environments with custom module paths, use the cluster-specific requirements:

```bash
pip install -r requirements_cluster.txt
```

### Environment Variables

R2P-GEN uses environment variables for flexible path configuration. Copy `.env.example` to `.env` and adjust as needed, or export them directly:

| Variable | Description | Default |
|---|---|---|
| `R2P_CLUSTER_MODE` | Enable cluster path layout (`true`/`false`) | `false` |
| `R2P_MODELS_BASE` | Base directory for locally cached model weights | *(HuggingFace Hub)* |
| `HF_HOME` | Override HuggingFace cache directory | *(HF default)* |
| `R2P_FLUX_MODEL` | Path or repo ID for the FLUX model | `black-forest-labs/FLUX.1-schnell` |
| `R2P_OUTPUT_DIR` | Output directory (cluster mode only) | `output/` |

When `R2P_MODELS_BASE` is not set, all models are downloaded automatically from HuggingFace Hub on first use.

---

## Dataset Preparation

R2P-GEN supports two standard personalization benchmarks:

- **DreamBench** — available at [dreambooth.github.io](https://dreambooth.github.io/)
- **PerVA** — available at [deepayan137.github.io/papers/training-free-personalization](https://deepayan137.github.io/papers/training-free-personalization.html)

Download and place datasets under `data/`. DreamBench can be prepared using the provided utility:

```bash
python prepare_dreambench.py
```

---

## Usage

### Step 1 — Start the FLUX Server

The FLUX model is kept resident in VRAM via a background HTTP server to avoid repeated loading across pipeline stages. Start it before any pipeline run:

```bash
python flux_server.py
```

The server listens on `http://127.0.0.1:8767` by default. Keep this process running in a separate terminal or tmux session throughout the experiment.

### Step 2 — Build the Fingerprint Database

```bash
python build_database.py
```

This runs Stage 0: CLIP-based view selection followed by Qwen3-VL fingerprint extraction. Output is written to `database/database.json`.

### Step 3 — Run the Pipeline

All pipeline stages are invoked through `flux_loop.py` via the `--stage` argument.

**Full automatic pipeline (recommended):**

```bash
python flux_loop.py \
    --database database/database.json \
    --output output/ \
    --stage full_auto
```

**Individual stages:**

```bash
# Stage 1: Generate images from fingerprints
python flux_loop.py --database database/database.json --output output/ --stage generate_only

# Stage 2: Verify generated images
python flux_loop.py --database database/database.json --output output/ --stage verify_base

# Stage 3: Refine rejected images
python flux_loop.py --database database/database.json --output output/ --stage refine

# Stage 4: Run final independent evaluation
python flux_loop.py --database database/database.json --output output/ --stage final_judge
```

**Distributed generation with sharding (for cluster use):**

```bash
# Run shard 0 of 4
python flux_loop.py \
    --database database/database.json \
    --output output/ \
    --stage generate_only \
    --num-shards 4 \
    --shard-index 0
```

### Ablation Studies

Two flags expose key architectural components for controlled ablation:

```bash
# Ablation 1: Naive prompt baseline (no fingerprints, no dense anchoring)
python flux_loop.py --database database/database.json --output output_naive/ \
    --stage generate_only --naive-prompt

# Ablation 2: Text-only generation (no image conditioning, T2I instead of I2I)
python flux_loop.py --database database/database.json --output output_t2i/ \
    --stage generate_only --no-image-cond
```

For end-to-end Slurm cluster runs, refer to the scripts in `scripts/`:

```bash
sbatch scripts/run_full_e2e.sh
```

---

## Evaluation

R2P-GEN uses a four-metric evaluation protocol computed by the **Final Judge** (InternVL3.5-8B), which operates independently from all prior pipeline stages to prevent self-evaluation bias.

| Metric | Model | Measures |
|---|---|---|
| **CLIP-I** | CLIP ViT-L/14 | Global visual identity similarity (generated vs. reference) |
| **CLIP-T** | CLIP ViT-L/14 | Text-image alignment (generated image vs. input prompt) |
| **DINO-I** | DINOv2 ViT-L/14 | Fine-grained structural and semantic fidelity |
| **TIFA Score** | InternVL3.5-8B (VQA) | Exact attribute verification via visual question answering |

Results are written to `output/final_judge_results.json`. Aggregate score analysis can be performed with:

```bash
python analyze_judge_scores.py --results output/final_judge_results.json
```

---

## Configuration

All hyperparameters and model paths are centralized in `config.py`. Key settings:

| Config class | Parameter | Default | Description |
|---|---|---|---|
| `Config.Generate` | `NUM_INFERENCE_STEPS` | `4` | FLUX denoising steps |
| `Config.Generate` | `BACKGROUND_STYLE` | `wooden_table` | Background template for generation |
| `Config.Refine` | `MAX_ITERATIONS` | `3` | Max refinement attempts per concept |
| `Config.Refine` | `TARGET_ACCURACY` | `0.95` | Target attribute verification score |
| `Config.Images` | `OUTPUT_IMAGE_SIZE` | `1024` | Output resolution (px) |

---

## Acknowledgements

This project was developed as part of the Foundation Models course for the Master's degree in Data Science at the University of Trento.
