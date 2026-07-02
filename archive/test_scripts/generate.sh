#!/bin/bash
#SBATCH --job-name=r2p_generate
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/generate/%j.out
#SBATCH --error=logs/generate/%j.err

echo "=========================================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi | head -20
echo "=========================================================="

# ===========================================================================
# 1. Env var (SPOSTATO IN ALTO)
# ===========================================================================
export R2P_PERVA_DATA=/leonardo_work/IscrC_MUSE/tballari/FM_Data/data/perva-data
export R2P_MODELS_BASE=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export R2P_CLUSTER_MODE=true
export HF_HOME=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export R2P_FLUX_MODEL=/leonardo_work/IscrC_MUSE/tballari/models_cache/FLUX.2-klein-9B
export R2P_OUTPUT_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0

# ===========================================================================
# 2. Setup
# ===========================================================================
module purge
module load profile/deeplrn
module load cuda/12.2
module load cudnn

cd /leonardo/home/userexternal/tballari/R2P-GEN
source $HOME/miniconda3/bin/activate FM_env

# ORA FUNZIONA CORRETTAMENTE
mkdir -p logs $R2P_OUTPUT_DIR/test_debug_time

# ===========================================================================
# 3. Generate
# ===========================================================================
echo "Avvio generate_only su database debug (5 concetti)..."

python -u flux_loop.py \
    --stage generate_only \
    --database database/database.json \
    --output $R2P_OUTPUT_DIR/test_debug

# ===========================================================================
# 4. Verifica output (CORRETTO L'USO DEL PATH)
# ===========================================================================
echo ""
echo "--- Output check ---"
echo "PNG generati:"
ls $R2P_OUTPUT_DIR/test_debug/*_generated.png 2>/dev/null && \
    ls $R2P_OUTPUT_DIR/test_debug/*_generated.png | wc -l | xargs echo "Totale:" || \
    echo "❌ Nessun PNG trovato"

echo ""
echo "Dimensioni file:"
ls -lh $R2P_OUTPUT_DIR/test_debug/*_generated.png 2>/dev/null || echo "(nessun file)"

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="