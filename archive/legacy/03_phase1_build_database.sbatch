#!/bin/bash
#SBATCH --job-name=r2p_build_db
#SBATCH --partition=edu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/build_db_%j.out
#SBATCH --error=logs/build_db_%j.err
#SBATCH --nodelist=edu02

echo "=== PHASE 1: BUILD DATABASE ==="
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
mkdir -p logs database output

# Build database from test split
python -u pipeline/build_database.py \
    --source_dir ~/data/perva-data/test \
    --output_path database/database_test.json \
    --device cuda

echo "Job finished at $(date)"