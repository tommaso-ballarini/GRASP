#!/bin/bash
#SBATCH --job-name=dreambench_test_e2e
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/dreambench_test_e2e/%j.out
#SBATCH --error=logs/dreambench_test_e2e/%j.err

# ===========================================================================
# Test END-TO-END self-contained su 3 concept di debug:
#   1) build_database_db.py   --debug --debug-limit 3   (ricrea il db da zero)
#   2) generate_dreambench.py                            (FLUX, zero-shot)
#   3) verify_dreambench.py                               (Qwen3-VL + CLIP)
#   4) refine_dreambench.py                                (recovery loop)
#
# GPU0: server FLUX (sempre attivo in background per generate + refine)
# GPU1: Qwen3-VL/CLIP, usato da build_database_db.py, verify e refine
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

cd /leonardo/home/userexternal/tballari/R2P-GEN
source $HOME/miniconda3/bin/activate FM_env

export PYTHONPATH=$PWD

export HF_HOME=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export RECOVERY_FLUX_URL="http://127.0.0.1:8766"

# FIX: R2P_PERVA_DATA è impostata globalmente sul cluster e punta al
# perva-data REALE (329 concept). Per questo test va sovrascritta
# esplicitamente con la cartella dreambench-data, altrimenti il build
# pesca dal dataset sbagliato.
export R2P_PERVA_DATA=/leonardo_work/IscrC_MUSE/tballari/FM_Data/dreambench-data

# Sanity check "fail-fast": se questa variabile fosse vuota o la cartella
# non esistesse, build_database_db.py cadrebbe silenziosamente sul default
# (perva-data REALE, 329 concept) perché usa `args.perva_data or DEFAULT`
# e una stringa vuota è "falsy" in Python. Meglio abortire subito con un
# errore chiaro che scoprirlo a posteriori nel database prodotto.
if [ -z "$R2P_PERVA_DATA" ] || [ ! -d "$R2P_PERVA_DATA" ]; then
    echo "❌ R2P_PERVA_DATA vuota o cartella inesistente: '$R2P_PERVA_DATA'. Abort."
    exit 1
fi
echo "✅ R2P_PERVA_DATA verificata: $R2P_PERVA_DATA"
echo "   Contenuto: $(ls "$R2P_PERVA_DATA/test" 2>/dev/null | tr '\n' ' ')"

# Path self-contained per questo test (separati dal run "vero")
TEST_DB_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_e2e
TEST_DATABASE=$TEST_DB_DIR/database_db_test.json
TEST_OUTPUT=$TEST_DB_DIR/output_dreambench_test

rm -rf "$TEST_OUTPUT"
mkdir -p logs/dreambench_test_e2e "$TEST_DB_DIR" "$TEST_OUTPUT"

echo "--- Sanity check ---"
python -c "from config import Config; Config.print_summary()"
echo "---------------------"

# ===========================================================================
# 1. Build database (debug, 3 concept) — usa GPU1 per Qwen3-VL/CLIP
# ===========================================================================
echo ""
echo "[1/4] Build database (debug, 3 concept)..."
echo "   Verifico R2P_PERVA_DATA: ${R2P_PERVA_DATA}"
CUDA_VISIBLE_DEVICES=1 python -u pipeline/build_database_db.py \
    --data-dir "$R2P_PERVA_DATA" \
    --split test \

# build_database_db.py salva sempre nello stesso path canonico
# (database/database_db.json, vedi Config.Database.CANONICAL_NAME). Lo
# copiamo nella cartella di test per non sovrascrivere il database "vero".
CANONICAL_DB="database/database_db.json"
if [ ! -f "$CANONICAL_DB" ]; then
    echo "❌ Database non trovato dopo il build → abort."
    exit 1
fi
cp "$CANONICAL_DB" "$TEST_DATABASE"
echo "✅ Database di test copiato in: $TEST_DATABASE"

# Sanity check: il database deve contenere path di dreambench-data,
# non di perva-data (altrimenti il bug R2P_PERVA_DATA si è ripresentato).
if grep -q "perva-data" "$TEST_DATABASE"; then
    echo "❌ Il database contiene path di perva-data, non dreambench-data! Abort."
    echo "   (verifica che --perva-data sia stato passato correttamente al build)"
    cat "$TEST_DATABASE" | head -20
    exit 1
fi
echo "✅ Sanity check OK: il database punta a dreambench-data."

# ===========================================================================
# 2. Avvio server FLUX (GPU0) — resta attivo per generate + refine
# ===========================================================================
echo ""
echo "[2/4] Avvio server FLUX su GPU0..."
CUDA_VISIBLE_DEVICES=0 python flux_server.py --port 8766 &
FLUX_PID=$!

echo "Attendo che il server FLUX sia pronto (polling su /health, timeout 180s)..."
FLUX_READY=0
for i in $(seq 1 36); do
    if curl -s -f -o /dev/null "http://127.0.0.1:8766/health"; then
        echo "✅ FLUX pronto dopo $((i * 5))s."
        FLUX_READY=1
        break
    fi
    sleep 5
done
if [ "$FLUX_READY" -eq 0 ]; then
    echo "❌ FLUX non raggiungibile dopo 180s. Abort."
    kill $FLUX_PID 2>/dev/null
    exit 1
fi

# ===========================================================================
# 3. Generate (zero-shot, 3 concept x 25 prompt x 4 immagini)
# ===========================================================================
echo ""
echo "[3/4] Generazione DreamBench (zero-shot)..."
CUDA_VISIBLE_DEVICES=1 python -u pipeline/generate_dreambench.py \
    --database "$TEST_DATABASE" \
    --output "$TEST_OUTPUT" \
    --images-per-prompt 4 \
    --batch-size 4

# ===========================================================================
# 4. Verify (esclude i 5 prompt property-mod) — GPU1
# ===========================================================================
echo ""
echo "[4/4a] Verify DreamBench..."
CUDA_VISIBLE_DEVICES=1 python -u pipeline/verify_dreambench.py \
    --database "$TEST_DATABASE" \
    --output "$TEST_OUTPUT"

# ===========================================================================
# 4b. Refine (recovery loop su rejected, subject_phrase only) — GPU1
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
echo "Termino il server FLUX (PID: $FLUX_PID)..."
kill $FLUX_PID

# ===========================================================================
# Output check
# ===========================================================================
echo ""
echo "--- Output check ---"
echo "Database di test: $TEST_DATABASE"
python -c "
import json
with open('$TEST_DATABASE') as f:
    db = json.load(f)
print(f'  Concetti nel db: {len(db.get(\"concept_dict\", {}))}')
"

echo ""
echo "Immagini generate:"
find "$TEST_OUTPUT" -name "*.png" | wc -l

echo ""
echo "Rejected (verify):"
if [ -f "$TEST_OUTPUT/rejected_dreambench.json" ]; then
    python -c "
import json
with open('$TEST_OUTPUT/rejected_dreambench.json') as f:
    rej = json.load(f)
print(f'  Immagini rejected: {len(rej)}')
"
else
    echo "  ⚠️  rejected_dreambench.json non trovato"
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
    echo "  ⚠️  recovery_results_dreambench.json non trovato (probabilmente nessun rejected)"
fi

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="
