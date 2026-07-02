#!/bin/bash
#SBATCH --job-name=r2p_full_e2e
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/full_e2e/%j.out
#SBATCH --error=logs/full_e2e/%j.err

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
mkdir -p logs/full_e2e

# ===========================================================================
# 2. Env var
# ===========================================================================
export R2P_PERVA_DATA=/leonardo_work/IscrC_MUSE/tballari/FM_Data/data/perva-data
export R2P_MODELS_BASE=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export R2P_CLUSTER_MODE=true
export HF_HOME=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export R2P_FLUX_MODEL=/leonardo_work/IscrC_MUSE/tballari/models_cache/FLUX.2-klein-9B

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Directory completamente dedicata a questo test — indipendente da test_100
OUTPUT_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_e2e
DATABASE=$OUTPUT_DIR/database_e2e.json
FLUX_PORT=8766

mkdir -p "$OUTPUT_DIR"

# ===========================================================================
# Cleanup: spegne il FLUX server anche in caso di interruzione
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
# STAGE 0 — BUILD DATABASE (10 concetti, GPU 0)
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 0/4 — BUILD DATABASE (debug_limit=10)"
echo "=========================================================="
START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python -u pipeline/build_database.py \
    --perva-data $R2P_PERVA_DATA \
    --debug \
    --debug-limit 10

END=$(date +%s)
echo "⏱️  Build completato in $((END - START))s"

# build_database.py scrive in database/database.json per default —
# copiamo nella output dir dedicata per tenere tutto isolato
cp database/database.json "$DATABASE"

if [ ! -f "$DATABASE" ]; then
    echo "❌ database_e2e.json non trovato — interruzione pipeline."
    exit 1
fi

python3 -c "
import json
with open('$DATABASE') as f:
    db = json.load(f)
print(f'   ✅ Concetti nel database: {len(db.get(\"concept_dict\", {}))}')
"

# ===========================================================================
# STAGE 1 — GENERATE (GPU 0)
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 1/4 — GENERATE"
echo "=========================================================="
START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python -u flux_loop.py \
    --stage generate_only \
    --database "$DATABASE" \
    --output   "$OUTPUT_DIR"

END=$(date +%s)
echo "⏱️  Generate completato in $((END - START))s"

PNG_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*_generated.png" | wc -l)
echo "   PNG generati: $PNG_COUNT"
if [ "$PNG_COUNT" -eq 0 ]; then
    echo "❌ Nessuna immagine generata — interruzione pipeline."
    exit 1
fi

# ===========================================================================
# STAGE 2 — VERIFY BASE (GPU 0)
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 2/4 — VERIFY BASE"
echo "=========================================================="
START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python -u flux_loop.py \
    --stage verify_base \
    --database "$DATABASE" \
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

# ===========================================================================
# STAGE 3 — REFINE (GPU 0: Qwen3 | GPU 1: FLUX server)
# ===========================================================================
if [ "$REJECTED_COUNT" -eq 0 ]; then
    echo "   ✅ Tutti i concetti hanno passato il verify. Refine saltato."
else
    echo ""
    echo "=========================================================="
    echo "STAGE 3/4 — REFINE (Qwen3 su GPU 0 | FLUX su GPU 1)"
    echo "=========================================================="

    echo "🚀 Avvio server FLUX in background sulla GPU 1 (porta $FLUX_PORT)..."
    CONDA_PYTHON=$WORK/tballari/envs/FM_env/bin/python
    CUDA_VISIBLE_DEVICES=1 $CONDA_PYTHON flux_server.py --port $FLUX_PORT \
        > "logs/full_e2e/flux_server_${SLURM_JOB_ID}.log" 2>&1 &
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
        echo "❌ FLUX Server non risponde dopo 10 minuti — interruzione pipeline."
        exit 1
    fi

    START=$(date +%s)

    CUDA_VISIBLE_DEVICES=0 python -u flux_loop.py \
        --stage refine \
        --database "$DATABASE" \
        --output   "$OUTPUT_DIR"

    END=$(date +%s)
    echo "⏱️  Refine completato in $((END - START))s"

    # Spegni il server FLUX — non serve più per il final_judge
    kill "$FLUX_PID" 2>/dev/null
    wait "$FLUX_PID" 2>/dev/null
    FLUX_PID=""
    echo "   🛑 Server FLUX spento."
fi

# ===========================================================================
# STAGE 4 — FINAL JUDGE (GPU 0)
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 4/4 — FINAL JUDGE"
echo "=========================================================="
START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python -u flux_loop.py \
    --stage final_judge \
    --database "$DATABASE" \
    --output   "$OUTPUT_DIR"

END=$(date +%s)
echo "⏱️  Final judge completato in $((END - START))s"

# ===========================================================================
# REPORT FINALE
# ===========================================================================
echo ""
echo "--- Report finale ---"
python3 -c "
import json, os

output_dir = '$OUTPUT_DIR'
database   = '$DATABASE'

with open(database) as f:
    db = json.load(f)
total = len(db.get('concept_dict', {}))

png_count = len([f for f in os.listdir(output_dir) if f.endswith('_generated.png')])
print(f'  Concetti database  : {total}')
print(f'  Immagini generate  : {png_count}')

rej_path = os.path.join(output_dir, 'rejected_concepts.json')
if os.path.exists(rej_path):
    with open(rej_path) as f:
        rej = json.load(f)
    print(f'  Verify passed      : {total - len(rej)}/{total}')
    print(f'  Verify rejected    : {len(rej)}/{total}')

rec_path = os.path.join(output_dir, 'recovery_results.json')
if os.path.exists(rec_path):
    with open(rec_path) as f:
        rec = json.load(f)
    recovered = sum(1 for v in rec.values() if v.get('status') == 'recovered')
    graveyard = sum(1 for v in rec.values() if v.get('status') == 'unrecoverable')
    print(f'  Refine recovered   : {recovered}')
    print(f'  Refine graveyard   : {graveyard}')

judge_path = os.path.join(output_dir, 'final_judge_results.json')
if os.path.exists(judge_path):
    with open(judge_path) as f:
        judge = json.load(f)
    print(f'  Judge valutati     : {len(judge)}')
    for cid, data in judge.items():
        m = data.get('metrics', {})
        print(f'   {cid}: CLIP-I={m.get(\"clip_i\",0):.3f} '
              f'| DINO-I={m.get(\"dino_i\",0):.3f} '
              f'| TIFA={m.get(\"tifa_score\",0):.1%}')
"

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="