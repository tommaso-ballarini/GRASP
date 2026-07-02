#!/bin/bash
#SBATCH --job-name=r2p_ablation_B
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ablation_B/%j.out
#SBATCH --error=logs/ablation_B/%j.err

echo "=========================================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi | head -20
echo "=========================================================="

module purge
module load cuda/12.2
module load cudnn

cd /leonardo/home/userexternal/tballari/R2P-GEN

CONDA_PYTHON=/leonardo_work/IscrC_MUSE/tballari/envs/FM_env/bin/python

mkdir -p logs/ablation_B

export R2P_PERVA_DATA=/leonardo_work/IscrC_MUSE/tballari/FM_Data/data/perva-data
export R2P_MODELS_BASE=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export R2P_CLUSTER_MODE=true
export HF_HOME=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export R2P_FLUX_MODEL=/leonardo_work/IscrC_MUSE/tballari/models_cache/FLUX.2-klein-9B

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUTPUT_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/ablation_full_B_text_fingerprints
DATABASE=database/database_centroid.json

mkdir -p "$OUTPUT_DIR"

echo "----------------------------------------------------------"
echo "STAGE 1 — GENERATE ONLY"
echo "----------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0 $CONDA_PYTHON -u flux_loop.py \
    --stage generate_only \
    --database "$DATABASE" \
    --output   "$OUTPUT_DIR"\
    --no-image-cond

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