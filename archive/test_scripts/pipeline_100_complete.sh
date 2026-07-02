#!/bin/bash
#SBATCH --job-name=r2p_pipeline_100
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=logs/pipeline_100/%j.out
#SBATCH --error=logs/pipeline_100/%j.err

# ===========================================================================
# 1. Setup
# ===========================================================================
echo "=========================================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi | head -20
echo "=========================================================="

module purge
module load profile/deeplrn
module load cuda/12.2
module load cudnn

cd /leonardo/home/userexternal/tballari/R2P-GEN
source $HOME/miniconda3/bin/activate FM_env

export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs/pipeline_100

# ===========================================================================
# 2. Env var
# ===========================================================================
export R2P_PERVA_DATA=/leonardo_work/IscrC_MUSE/tballari/FM_Data/data/perva-data
export R2P_MODELS_BASE=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export R2P_CLUSTER_MODE=true
export HF_HOME=/leonardo_work/IscrC_MUSE/tballari/models_cache/.hf_cache
export R2P_FLUX_MODEL=/leonardo_work/IscrC_MUSE/tballari/models_cache/FLUX.2-klein-9B

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUTPUT_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_100
FLUX_PORT=8766
DATABASE_JSON="database/database.json"

mkdir -p "$OUTPUT_DIR"

# ===========================================================================
# Funzione cleanup: garantisce che il server FLUX venga spento anche se il
# job viene interrotto (SIGTERM da SLURM, errore, Ctrl-C in test locale).
# ===========================================================================
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

# ===========================================================================
# STAGE 1 — BUILD DATABASE (100 concetti, GPU 0)
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 1/4 — BUILD DATABASE (100 concetti)"
echo "=========================================================="
START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python -u pipeline/build_dataset.py \
    --split train \
    --debug \
    --debug-limit 100

END=$(date +%s)
echo "⏱️  Build completato in $((END - START))s"

if [ ! -f "$DATABASE_JSON" ]; then
    echo "❌ $DATABASE_JSON non trovato — interruzione pipeline."
    exit 1
fi

python3 -c "
import json
with open('$DATABASE_JSON') as f:
    db = json.load(f)
n = len(db.get('concept_dict', {}))
print(f'   ✅ Concetti nel database: {n}')
"

# ===========================================================================
# STAGE 2 — GENERATE (GPU 0)
# Il FLUX server NON è ancora attivo: generate.py usa la pipe in-process.
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 2/4 — GENERATE"
echo "=========================================================="
START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python -u flux_loop.py \
    --stage generate_only \
    --database "$DATABASE_JSON" \
    --output   "$OUTPUT_DIR"

END=$(date +%s)
echo "⏱️  Generate completato in $((END - START))s"

PNG_COUNT=$(ls "$OUTPUT_DIR"/*_generated.png 2>/dev/null | wc -l)
echo "   PNG generati: $PNG_COUNT"

if [ "$PNG_COUNT" -eq 0 ]; then
    echo "❌ Nessuna immagine generata — interruzione pipeline."
    exit 1
fi

# ===========================================================================
# STAGE 3 — VERIFY BASE (GPU 0, Qwen3-VL + CLIP)
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 3/4 — VERIFY BASE (Qwen3-VL + CLIP)"
echo "=========================================================="
START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python -u flux_loop.py \
    --stage verify_base \
    --database "$DATABASE_JSON" \
    --output   "$OUTPUT_DIR"

END=$(date +%s)
echo "⏱️  Verify completato in $((END - START))s"

REJECTED_PATH="$OUTPUT_DIR/rejected_concepts.json"
if [ ! -f "$REJECTED_PATH" ]; then
    echo "❌ rejected_concepts.json non trovato — interruzione pipeline."
    exit 1
fi

REJECTED_COUNT=$(python3 -c "
import json
with open('$REJECTED_PATH') as f:
    r = json.load(f)
print(len(r))
")

echo "   Concetti rejected: $REJECTED_COUNT"

if [ "$REJECTED_COUNT" -eq 0 ]; then
    echo "   ✅ Tutti i concetti hanno passato la verifica. Refine saltato."
    # Salta direttamente al report finale
else
    # ===========================================================================
    # STAGE 4 — REFINE (GPU 0: Qwen3-VL | GPU 1: FLUX server)
    # Il server FLUX viene avviato qui, subito prima del refine, per non
    # tenere occupata la GPU 1 durante build/generate/verify.
    # ===========================================================================
    echo ""
    echo "=========================================================="
    echo "STAGE 4/4 — REFINE (Qwen3 su GPU 0 | FLUX server su GPU 1)"
    echo "=========================================================="

    echo "🚀 Avvio server FLUX in background sulla GPU 1 (porta $FLUX_PORT)..."
    CUDA_VISIBLE_DEVICES=1 python flux_server.py --port $FLUX_PORT \
        > "logs/pipeline_100/flux_server_${SLURM_JOB_ID}.log" 2>&1 &
    FLUX_PID=$!

    echo "⏳ Attendo che il server FLUX sia pronto (max 10 min)..."
    FLUX_READY=0
    for i in $(seq 1 60); do
        if curl -s "http://127.0.0.1:$FLUX_PORT/health" > /dev/null 2>&1; then
            echo "   ✅ FLUX Server pronto! (dopo ~$((i * 10))s)"
            FLUX_READY=1
            break
        fi
        sleep 10
    done

    if [ "$FLUX_READY" -eq 0 ]; then
        echo "❌ FLUX Server non risponde dopo 10 minuti. Abort."
        exit 1
    fi

    START=$(date +%s)

    CUDA_VISIBLE_DEVICES=0 python -u flux_loop.py \
        --stage refine \
        --database "$DATABASE_JSON" \
        --output   "$OUTPUT_DIR"

    END=$(date +%s)
    echo "⏱️  Refine completato in $((END - START))s"

    # Il trap cleanup() spegne il server FLUX all'uscita dello script.
fi

# ===========================================================================
# REPORT FINALE
# ===========================================================================
echo ""
echo "=========================================================="
echo "REPORT FINALE"
echo "=========================================================="

python3 -c "
import json, os

database_path = '$DATABASE_JSON'
output_dir    = '$OUTPUT_DIR'

with open(database_path) as f:
    db = json.load(f)
total = len(db.get('concept_dict', {}))

png_count = len([f for f in os.listdir(output_dir) if f.endswith('_generated.png')])

print(f'  Concetti nel database : {total}')
print(f'  Immagini generate     : {png_count}')

rejected_path = os.path.join(output_dir, 'rejected_concepts.json')
if os.path.exists(rejected_path):
    with open(rejected_path) as f:
        rejected = json.load(f)
    passed_verify = total - len(rejected)
    print(f'  Verify passed        : {passed_verify}/{total}')
    print(f'  Verify rejected      : {len(rejected)}/{total}')

recovery_path = os.path.join(output_dir, 'recovery_results.json')
if os.path.exists(recovery_path):
    with open(recovery_path) as f:
        recovery = json.load(f)
    recovered   = sum(1 for v in recovery.values() if v.get('status') == 'recovered')
    unrecovered = sum(1 for v in recovery.values() if v.get('status') == 'unrecoverable')
    print(f'  Refine recovered     : {recovered}')
    print(f'  Refine graveyard     : {unrecovered}')
    final_passed = passed_verify + recovered if os.path.exists(rejected_path) else recovered
    print(f'  --- Totale approvati : {final_passed}/{total} ---')
else:
    print('  (recovery_results.json non trovato — refine non eseguito)')
"

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="