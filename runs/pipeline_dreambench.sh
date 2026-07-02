#!/bin/bash
#SBATCH --job-name=r2p_ablation_A
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --partition=<YOUR_SLURM_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ablation_A/%j.out
#SBATCH --error=logs/ablation_A/%j.err

# ===========================================================================
# Test END-TO-END self-contained on 3 concept for debugging:
#   1) build_database_db.py   --debug --debug-limit 3       (recreate db)
#   2) generate_dreambench.py                               (FLUX, zero-shot)
#   3) verify_dreambench.py                                 (Qwen3-VL + CLIP)
#   4) refine_dreambench.py                                 (recovery loop)
#
# GPU0: server FLUX (always active in background for generate + refine)
# GPU1: Qwen3-VL/CLIP, used for build_database_db.py, verify and refine
# ===========================================================================

echo "=========================================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi
echo "=========================================================="

module purge
module load profile/deeplrn
module load cuda/12.2
module load cudnn

cd <YOUR_PROJECT_DIR>
source <YOUR_CONDA_BASE>/bin/activate FM_env

export PYTHONPATH=$PWD

export HF_HOME=<YOUR_HF_CACHE_DIR>
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export RECOVERY_FLUX_URL="http://127.0.0.1:8766"

export R2P_PERVA_DATA=<YOUR_DREAMBENCH_DATA_DIR>

### SANITY CHECK: R2P_PERVA_DATA NEED TO POINT TO DREAMBENCH-DATA, NOT PERVA-DATA ###
if [ -z "$R2P_PERVA_DATA" ] || [ ! -d "$R2P_PERVA_DATA" ]; then
    echo "❌ R2P_PERVA_DATA empty or directory does not exist: '$R2P_PERVA_DATA'. Abort."
    exit 1
fi
echo "✅ R2P_PERVA_DATA verified: $R2P_PERVA_DATA"
echo "   Content: $(ls "$R2P_PERVA_DATA/test" 2>/dev/null | tr '\n' ' ')"

TEST_DB_DIR=<YOUR_OUTPUT_DIR>/test_e2e
TEST_DATABASE=$TEST_DB_DIR/database_db_test.json
TEST_OUTPUT=$TEST_DB_DIR/output_dreambench_test

rm -rf "$TEST_OUTPUT"
mkdir -p logs/dreambench_test_e2e "$TEST_DB_DIR" "$TEST_OUTPUT"

echo "--- Sanity check ---"
python -c "from config import Config; Config.print_summary()"
echo "---------------------"

# ===========================================================================
# 1. Build database (debug, 3 concept)
# ===========================================================================
echo ""
echo "[1/4] Build database (debug, 3 concept)..."
echo "   Verifying R2P_PERVA_DATA: ${R2P_PERVA_DATA}"
CUDA_VISIBLE_DEVICES=1 python -u pipeline/build_database_db.py \
    --data-dir "$R2P_PERVA_DATA" \
    --split test \



CANONICAL_DB="database/database_db.json"
if [ ! -f "$CANONICAL_DB" ]; then
    echo "❌ Database not found after build → abort."
    exit 1
fi
cp "$CANONICAL_DB" "$TEST_DATABASE"
echo "✅ Test database copied to: $TEST_DATABASE"

# Sanity check: the database must contain paths to dreambench-data, not perva-data

if grep -q "perva-data" "$TEST_DATABASE"; then
    echo "❌ The database contains paths to perva-data, not dreambench-data! Abort."
    echo "   (verify that --perva-data was passed correctly to the build)"
    cat "$TEST_DATABASE" | head -20
    exit 1
fi
echo "✅ Sanity check OK: the database points to dreambench-data."

# ===========================================================================
# 2. Starting FLUX server (GPU0) — remains active for generate + refine
# ===========================================================================
echo ""
echo "[2/4] Starting FLUX server on GPU0..."
CUDA_VISIBLE_DEVICES=0 python flux_server.py --port 8766 &
FLUX_PID=$!

echo "Waiting for FLUX server to be ready (polling on /health, timeout 180s)..."
FLUX_READY=0
for i in $(seq 1 36); do
    if curl -s -f -o /dev/null "http://127.0.0.1:8766/health"; then
        echo "✅ FLUX ready after $((i * 5))s."
        FLUX_READY=1
        break
    fi
    sleep 5
done
if [ "$FLUX_READY" -eq 0 ]; then
    echo "❌ FLUX non available after 180s. Abort."
    kill $FLUX_PID 2>/dev/null
    exit 1
fi

# ===========================================================================
# 3. Generate (zero-shot, 3 concept x 25 prompt x 4 immagini)
# ===========================================================================
echo ""
echo "[3/4] Generating DreamBench (zero-shot)..."
CUDA_VISIBLE_DEVICES=1 python -u pipeline/generate_dreambench.py \
    --database "$TEST_DATABASE" \
    --output "$TEST_OUTPUT" \
    --images-per-prompt 4 \
    --batch-size 4

# ===========================================================================
# 4. Verify — GPU1
# ===========================================================================
echo ""
echo "[4/4a] Verify DreamBench..."
CUDA_VISIBLE_DEVICES=1 python -u pipeline/verify_dreambench.py \
    --database "$TEST_DATABASE" \
    --output "$TEST_OUTPUT"

# ===========================================================================
# 4b. Refine (recovery loop over rejected images, subject_phrase only) — GPU1
# ===========================================================================
echo ""
echo "[4/4b] Refine DreamBench..."
CUDA_VISIBLE_DEVICES=1 python -u pipeline/refine_dreambench.py \
    --database "$TEST_DATABASE" \
    --output "$TEST_OUTPUT"

# ===========================================================================
# Cleanup
# ===========================================================================
echo ""
echo "Closing the FLUX server (PID: $FLUX_PID)..."
kill $FLUX_PID

# ===========================================================================
# Output check
# ===========================================================================
echo ""
echo "--- Output check ---"
echo "Test database: $TEST_DATABASE"
python -c "
import json
with open('$TEST_DATABASE') as f:
    db = json.load(f)
print(f'  Concepts in db: {len(db.get(\"concept_dict\", {}))}')
"

echo ""
echo "Images generated:"
find "$TEST_OUTPUT" -name "*.png" | wc -l

echo ""
echo "Rejected (verify):"
if [ -f "$TEST_OUTPUT/rejected_dreambench.json" ]; then
    python -c "
import json
with open('$TEST_OUTPUT/rejected_dreambench.json') as f:
    rej = json.load(f)
print(f'  Images rejected: {len(rej)}')
"
else
    echo "  ⚠️  rejected_dreambench.json not found"
fi

echo ""
echo "Recovery (refine):"
if [ -f "$TEST_OUTPUT/recovery_results_dreambench.json" ]; then
    python -c "
import json
with open('$TEST_OUTPUT/recovery_results_dreambench.json') as f:
    rec = json.load(f)
recovered = sum(1 for v in rec.values() if v['status'] == 'recovered')
graveyard = sum(1 for v in rec.values() if v['status'] == 'unrecoverable')
print(f'  Recovered: {recovered} | Graveyard: {graveyard}')
for k, v in list(rec.items())[:3]:
    print(f'  [{k}] status={v[\"status\"]}')
"
else
    echo "  ⚠️  recovery_results_dreambench.json not found (probably no rejected images)"
fi

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="
