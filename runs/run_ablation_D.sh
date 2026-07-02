#!/bin/bash
#SBATCH --job-name=r2p_ablation_A
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --partition=<YOUR_SLURM_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ablation_A/%j.out
#SBATCH --error=logs/ablation_A/%j.err

echo "=========================================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi | head -20
echo "=========================================================="

module purge
module load cuda/12.2
module load cudnn

cd <YOUR_PROJECT_DIR>

CONDA_PYTHON=<YOUR_CONDA_ENV_PYTHON>

mkdir -p logs/ablation_D

export R2P_PERVA_DATA=<YOUR_PERVA_DATA_DIR>
export R2P_MODELS_BASE=<YOUR_HF_CACHE_DIR>
export R2P_CLUSTER_MODE=true
export HF_HOME=<YOUR_HF_CACHE_DIR>
export R2P_FLUX_MODEL=<YOUR_FLUX_MODEL_DIR>
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUTPUT_DIR=<YOUR_OUTPUT_DIR>/ablation_full_D_image_fingerprints_norefine
DATABASE=database/database_centroid.json

mkdir -p "$OUTPUT_DIR"

echo "----------------------------------------------------------"
echo "STAGE 1 — GENERATE ONLY"
echo "----------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0 $CONDA_PYTHON -u flux_loop.py \
    --stage generate_only \
    --database "$DATABASE" \
    --output   "$OUTPUT_DIR"

echo "----------------------------------------------------------"
echo "STAGE 2 — VERIFY BASE"
echo "----------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0 $CONDA_PYTHON -u flux_loop.py \
    --stage verify_base \
    --database "$DATABASE" \
    --output   "$OUTPUT_DIR"

echo "----------------------------------------------------------"
echo "STAGE 4 — FINAL JUDGE"
echo "----------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0 $CONDA_PYTHON -u flux_loop.py \
    --stage final_judge \
    --database "$DATABASE" \
    --output   "$OUTPUT_DIR"

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="