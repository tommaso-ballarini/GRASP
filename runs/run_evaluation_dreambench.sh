#!/bin/bash
#SBATCH --job-name=dreambench_eval
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dreambench_eval/%j.out
#SBATCH --error=logs/dreambench_eval/%j.err
# ===========================================================================
# Fase 4 — Valutazione DreamBench (DINO-I / CLIP-I / CLIP-T ufficiali).
#
# Valuta SOLO le immagini finali in $TEST_OUTPUT (già "full": originali
# per non-rejected/graveyard, sovrascritte per recovered — vedi
# refine_dreambench.py). Nessun bisogno di backup zero-shot separato.
#
# PREREQUISITO: evaluate_dreambench.py::_find_generated_images deve
# escludere i residui "*_rejected_attempt*.png" lasciati da refine
# (altrimenti le immagini recovered vengono contate/mediate due volte).
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

TEST_DB_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_e2e
TEST_DATABASE=$TEST_DB_DIR/database_db_test.json
TEST_OUTPUT=$TEST_DB_DIR/output_dreambench_test

RESULTS_DIR=$TEST_DB_DIR/dreambench_metrics
mkdir -p logs/dreambench_eval "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Sanity check 1: database esiste ed è quello giusto (dreambench, non perva)
# ---------------------------------------------------------------------------
if [ ! -f "$TEST_DATABASE" ]; then
    echo "❌ Database di test non trovato: $TEST_DATABASE"
    exit 1
fi
if grep -q "perva-data" "$TEST_DATABASE"; then
    echo "❌ Il database contiene path di perva-data, non dreambench-data! Abort."
    exit 1
fi
echo "✅ Database OK: $TEST_DATABASE"

# ---------------------------------------------------------------------------
# Sanity check 2: checkpoint HF offline presenti (CLIP-B/32 + DINO-vits16)
# ---------------------------------------------------------------------------
echo "--- Sanity check modelli DreamBench ---"
python -c "
from config import Config
import os
for name in ['CLIP_DREAMBENCH_MODEL', 'DINO_DREAMBENCH_MODEL']:
    path = getattr(Config.Models, name)
    ok = os.path.isdir(path)
    print(f'  {name}: {path}  ->  {\"OK\" if ok else \"MANCANTE\"}')
    if not ok:
        raise SystemExit(f'Checkpoint mancante: {path}')
"
if [ $? -ne 0 ]; then
    echo "❌ Checkpoint HF mancanti. Con HF_HUB_OFFLINE=1 il job fallirebbe."
    exit 1
fi
echo "✅ Checkpoint presenti."

# ---------------------------------------------------------------------------
# Sanity check 3: immagini reali dei concept NON puntano a perva-data
# ---------------------------------------------------------------------------
python -c "
import json
with open('$TEST_DATABASE') as f:
    db = json.load(f)
bad = []
for cid, c in db.get('concept_dict', {}).items():
    imgs = c.get('image', [])
    if isinstance(imgs, str):
        imgs = [imgs]
    for p in imgs:
        if 'perva-data' in p:
            bad.append((cid, p))
if bad:
    print('❌ Immagini reali con path perva-data trovate:')
    for cid, p in bad[:10]:
        print(f'   {cid}: {p}')
    raise SystemExit(1)
print('✅ Tutte le immagini reali puntano a dreambench-data.')
"
if [ $? -ne 0 ]; then
    exit 1
fi

# ---------------------------------------------------------------------------
# Sanity check 4 (info, non blocca): quanti residui _rejected_attempt* ci sono
# ---------------------------------------------------------------------------
N_RESIDUI=$(find "$TEST_OUTPUT" -name "*_rejected_attempt*.png" | wc -l)
echo "ℹ️  Residui _rejected_attempt*.png trovati in output: $N_RESIDUI"
echo "   (devono essere esclusi dal glob in _find_generated_images — verifica che il fix sia applicato)"

# ===========================================================================
# Valutazione FULL (immagine finale per ogni concept/prompt/img_idx)
# ===========================================================================
echo ""
echo "[EVAL] Full (immagini finali post-refine): $TEST_OUTPUT"
CUDA_VISIBLE_DEVICES=0 python -u pipeline/evaluate_dreambench.py \
    --database "$TEST_DATABASE" \
    --output "$TEST_OUTPUT" \
    --results-dir "$RESULTS_DIR/full" \
    --label "R2P-GEN (full)" \
    --device cuda

echo ""
echo "=========================================================="
echo "Risultati in: $RESULTS_DIR/full/metrics_table.txt"
echo "Job finished at $(date)"
echo "=========================================================="