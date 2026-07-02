#!/bin/bash
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --job-name=R2P_Flux_Gen
#SBATCH --output=logs/flux_gen/%j.out
#SBATCH --error=logs/flux_gen/%j.err

# ===========================================================================
# 1. Moduli HPC (Leonardo)
# ===========================================================================
module purge
module load profile/deeplrn
module load cuda/12.2   # 12.2 invece di 12.1: nvcc disponibile, compatibile con torch 2.5.1+cu121
module load cudnn

# ===========================================================================
# 2. Ambiente Conda
# ===========================================================================
cd /leonardo_work/IscrC_MUSE/tballari/R2P-GEN
source $HOME/miniconda3/bin/activate FM_env

# ===========================================================================
# 3. Variabili d'Ambiente
# ===========================================================================

# --- HuggingFace ---
# HF_HOME punta alla cache modelli, NON alla cartella huggingface dei pesi.
# I pesi stanno in $R2P_MODELS_BASE, la cache (.hf_cache) in $HF_HOME.
export HF_HOME="/leonardo_work/IscrC_MUSE/tballari/models_cache/.hf_cache"
export R2P_MODELS_BASE="/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface"

# Offline mode: blocca ogni tentativo di download (fondamentale sul compute node)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- Cluster mode per config.py ---
export R2P_CLUSTER_MODE=true

# --- PyTorch ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- Flash Attention: usa l'implementazione installata ---
# (flash_attn 2.7.3 installata in FM_env, wheel cxx11abiFALSE)
# Non serve export aggiuntivo: viene rilevata automaticamente da transformers.

# ===========================================================================
# 4. Paths pipeline
# ===========================================================================
DATABASE_JSON="database/database.json"
OUTPUT_DIR="output/generazioni_flux"
NUM_GPUS=4

mkdir -p logs/flux_gen
mkdir -p "$OUTPUT_DIR"

# ===========================================================================
# 5. Esecuzione parallela (1 processo per GPU, sharding via flux_loop.py)
# ===========================================================================
echo "=========================================================="
echo "🚀 INIZIO GENERAZIONE PARALLELA SU $NUM_GPUS GPU"
echo "   Database : $DATABASE_JSON"
echo "   Output   : $OUTPUT_DIR"
echo "   HF_HOME  : $HF_HOME"
echo "   Models   : $R2P_MODELS_BASE"
echo "=========================================================="

for ((i=0; i<NUM_GPUS; i++)); do
    echo "   Lancio Worker $i su GPU fisica $i (shard $i/$NUM_GPUS)..."
    CUDA_VISIBLE_DEVICES=$i python flux_loop.py \
        --stage      generate_only \
        --database   "$DATABASE_JSON" \
        --output     "$OUTPUT_DIR" \
        --num-shards "$NUM_GPUS" \
        --shard-index "$i" \
        > "logs/flux_gen/${SLURM_JOB_ID}_worker${i}.log" 2>&1 &
done

# Aspetta tutti i worker
wait

echo "=========================================================="
echo "✅ Generazione parallela completata."
echo "   Log per worker: logs/flux_gen/${SLURM_JOB_ID}_worker*.log"
echo "=========================================================="

# ===========================================================================
# 6. Verify + Recovery + Final Judge (sequenziale, dopo che tutti i worker
#    hanno finito — usa 1 GPU sola, le altre vengono liberate)
# ===========================================================================
echo ""
echo "📍 Avvio VERIFY BASE..."
CUDA_VISIBLE_DEVICES=0 python flux_loop.py \
    --stage    verify_base \
    --database "$DATABASE_JSON" \
    --output   "$OUTPUT_DIR"

echo ""
echo "🚑 Avvio RECOVERY..."
CUDA_VISIBLE_DEVICES=0 python flux_loop.py \
    --stage    recovery \
    --database "$DATABASE_JSON" \
    --output   "$OUTPUT_DIR"

echo ""
echo "⚖️  Avvio FINAL JUDGE..."
CUDA_VISIBLE_DEVICES=0 python flux_loop.py \
    --stage    final_judge \
    --database "$DATABASE_JSON" \
    --output   "$OUTPUT_DIR"

echo ""
echo "🏁 PIPELINE COMPLETATA — risultati in $OUTPUT_DIR"