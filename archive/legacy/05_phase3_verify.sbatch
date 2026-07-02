#!/bin/bash
#SBATCH --job-name=r2p_verify
#SBATCH --partition=edu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#SBATCH --nodelist=edu02

echo "=== PHASE 3: VERIFY IMAGES ==="
echo "Job started at $(date)"
echo "Node: $(hostname)"
nvidia-smi

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

source ~/venvs/r2p-gen/bin/activate
export R2P_CLUSTER_MODE=true
export CUDA_VISIBLE_DEVICES=0

cd ~/R2P-GEN

# Verify only, skip generation (uses existing images)
python -u full_loop.py \
    --mode database \
    --database database/database_test.json \
    --output output/generated \
    --skip-generation

echo "Job finished at $(date)"