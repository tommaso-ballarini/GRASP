#!/bin/bash
#SBATCH --job-name=r2p_ablation_E
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/ablation_E/%j.out
#SBATCH --error=logs/ablation_E/%j.err

echo "=========================================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi | head -20
echo "=========================================================="

module purge
# Rimosso il modulo profile/deeplrn per non corrompere l'ambiente Conda
module load cuda/12.2
module load cudnn

cd /leonardo/home/userexternal/tballari/R2P-GEN

# Usiamo il path assoluto di python del tuo ambiente Conda in modo coerente per TUTTO
CONDA_PYTHON=/leonardo_work/IscrC_MUSE/tballari/envs/FM_env/bin/python

mkdir -p logs/ablation_E

export R2P_PERVA_DATA=/leonardo_work/IscrC_MUSE/tballari/FM_Data/data/perva-data
export R2P_MODELS_BASE=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export R2P_CLUSTER_MODE=true
export HF_HOME=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export R2P_FLUX_MODEL=/leonardo_work/IscrC_MUSE/tballari/models_cache/FLUX.2-klein-9B

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUTPUT_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/ablation_full_E_full_centroid
DATABASE=database/database_centroid.json
FLUX_PORT=8766

mkdir -p "$OUTPUT_DIR"

FLUX_PID=""
cleanup() {
    if [ -n "$FLUX_PID" ]; then
        echo "🛑 Chiudo server FLUX (PID=$FLUX_PID)..."
        kill "$FLUX_PID" 2>/dev/null
        wait "$FLUX_PID" 2>/dev/null
        echo "   Server FLUX terminato."
    fi
}
trap cleanup EXIT

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

REJECTED_PATH="$OUTPUT_DIR/rejected_concepts.json"

if [ ! -f "$REJECTED_PATH" ]; then
    echo "❌ CRASH RILEVATO: rejected_concepts.json non è stato generato dallo stage verify. Interruzione."
    exit 1
fi

REJECTED_COUNT=$(python3 -c "import json; print(len(json.load(open('$REJECTED_PATH'))))")

if [ "$REJECTED_COUNT" -eq 0 ]; then
    echo "   ✅ Tutti i concetti hanno passato il verify. Refine saltato."
else
    echo "----------------------------------------------------------"
    echo "STAGE 3 — REFINE (Qwen3 su GPU 0 | FLUX su GPU 1)"
    echo "----------------------------------------------------------"
    echo "🚀 Avvio server FLUX in background sulla GPU 1 (porta $FLUX_PORT)..."
    
    CUDA_VISIBLE_DEVICES=1 $CONDA_PYTHON flux_server.py --port $FLUX_PORT \
        > "logs/ablation_E/flux_server_${SLURM_JOB_ID}.log" 2>&1 &
    FLUX_PID=$!

    echo "⏳ Attendo che il server FLUX sia pronto..."
    FLUX_READY=0
    for i in $(seq 1 60); do
        if curl -s "http://127.0.0.1:$FLUX_PORT/health" > /dev/null 2>&1; then
            echo "   ✅ FLUX Server pronto!"
            FLUX_READY=1
            break
        fi
        sleep 10
    done

    if [ "$FLUX_READY" -eq 0 ]; then
        echo "❌ FLUX Server non risponde — interruzione pipeline."
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=0 $CONDA_PYTHON -u flux_loop.py \
        --stage refine \
        --database "$DATABASE" \
        --output   "$OUTPUT_DIR"

    kill "$FLUX_PID" 2>/dev/null
    wait "$FLUX_PID" 2>/dev/null
    FLUX_PID=""
    echo "   🛑 Server FLUX spento."
fi

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