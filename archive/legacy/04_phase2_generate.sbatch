#!/bin/bash
#SBATCH --job-name=r2p_generate
#SBATCH --partition=edu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --nodelist=edu02

echo "=== PHASE 2: GENERATE IMAGES ==="
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
mkdir -p output/generated

# Generate only, skip verification
python -u full_loop.py \
    --mode database \
    --database database/database_test.json \
    --output output/generated \
    --skip-verification

echo "Job finished at $(date)"