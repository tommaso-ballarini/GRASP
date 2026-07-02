#!/bin/bash
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2       # <--- 2 GPU richieste
#SBATCH --mem=128GB
#SBATCH --job-name=R2P_Refine
#SBATCH --output=logs/pipeline_40/refine_%j.out
#SBATCH --error=logs/pipeline_40/refine_%j.err

# ==========================================
# SETUP HPC E CONDA
# ==========================================
module purge
module load profile/deeplrn
module load cuda/12.2
module load cudnn

cd /leonardo/home/userexternal/tballari/R2P-GEN
source $HOME/miniconda3/bin/activate FM_env

# Esporta il path per far trovare config.py
export PYTHONPATH=$PWD:$PYTHONPATH

# ==========================================
# VARIABILI AMBIENTE (Offline & Paths)
# ==========================================
export HF_HOME="/leonardo_work/IscrC_MUSE/tballari/models_cache/.hf_cache"
export R2P_MODELS_BASE="/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

DATABASE_JSON="database/database_buono.json"
OUTPUT_DIR="/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_40"
FLUX_PORT=8766

# Funzione per pulire i background process se il job viene killato
cleanup() {
    if [ -n "$FLUX_PID" ]; then
        echo "🛑 Chiudo server FLUX (PID=$FLUX_PID)"
        kill $FLUX_PID 2>/dev/null
        wait $FLUX_PID 2>/dev/null
    fi
}
trap cleanup EXIT

# ==========================================
# 1. AVVIO SERVER FLUX (Su GPU 1)
# ==========================================
echo "🚀 Avvio server FLUX in background sulla porta $FLUX_PORT (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 python flux_server.py --port $FLUX_PORT > logs/pipeline_50/flux_server_${SLURM_JOB_ID}.log 2>&1 &
FLUX_PID=$!

echo "⏳ Attendo che FLUX sia pronto..."
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:$FLUX_PORT/health" > /dev/null 2>&1; then
        echo "   ✅ FLUX Server pronto!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "   ❌ FLUX Server non risponde dopo 10 minuti. Abort."
        exit 1
    fi
    sleep 10
done

# ==========================================
# 2. AVVIO REFINE LOOP (Su GPU 0 con Qwen3)
# ==========================================
echo "⚖️ Avvio Pipeline Refine (GPU 0)..."

CUDA_VISIBLE_DEVICES=0 python flux_loop.py \
    --stage refine \
    --database $DATABASE_JSON \
    --output $OUTPUT_DIR

echo "🏁 Job Refine Completato!"