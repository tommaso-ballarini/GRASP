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

# DreamBench Evaluation(DINO-I / CLIP-I / CLIP-T).
#
# Evaluates ONLY the final images in $TEST_OUTPUT (already "full": original
# for non-rejected/graveyard, overwritten for recovered, see
# refine_dreambench.py). No need for a separate zero-shot backup.

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

TEST_DB_DIR=<YOUR_OUTPUT_DIR>/test_e2e
TEST_DATABASE=$TEST_DB_DIR/database_db_test.json
TEST_OUTPUT=$TEST_DB_DIR/output_dreambench_test

RESULTS_DIR=$TEST_DB_DIR/dreambench_metrics
mkdir -p logs/dreambench_eval "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Sanity check 1: database needs to exist and be the correct one (dreambench, not perva)
# ---------------------------------------------------------------------------
if [ ! -f "$TEST_DATABASE" ]; then
    echo "❌ Testing Database not found: $TEST_DATABASE"
    exit 1
fi
if grep -q "perva-data" "$TEST_DATABASE"; then
    echo "❌ The database contains paths to perva-data, not dreambench-data! Abort."
    exit 1
fi
echo "✅ Database OK: $TEST_DATABASE"

# ---------------------------------------------------------------------------
# Sanity check 2:  HF offline checkpoints presents? (CLIP-B/32 + DINO-vits16)
# ---------------------------------------------------------------------------
echo "--- Sanity check DreamBench models ---"
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
    echo "❌ HF checkpoints missing. With HF_HUB_OFFLINE=1 the job would fail."
    exit 1
fi
echo "✅ Checkpoint presents."

# ---------------------------------------------------------------------------
# Sanity check 3: real concept images do not have to point to perva-data
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
    print('❌ Real concept images with path perva-data found:')
    for cid, p in bad[:10]:
        print(f'   {cid}: {p}')
    raise SystemExit(1)
print('✅ All real concept images point to dreambench-data.')
"
if [ $? -ne 0 ]; then
    exit 1
fi

# ---------------------------------------------------------------------------
# Sanity check 4 (info, does not block): how many residual _rejected_attempt* are there?
# ---------------------------------------------------------------------------
N_RESIDUI=$(find "$TEST_OUTPUT" -name "*_rejected_attempt*.png" | wc -l)
echo "ℹ️  Residui _rejected_attempt*.png trovati in output: $N_RESIDUI"
echo "   (devono essere esclusi dal glob in _find_generated_images — verifica che il fix sia applicato)"

# ===========================================================================
# FULL evaluation (final image for each concept/prompt/img_idx)
# ===========================================================================
echo ""
echo "[EVAL] Full (post-refine final images): $TEST_OUTPUT"
CUDA_VISIBLE_DEVICES=0 python -u pipeline/evaluate_dreambench.py \
    --database "$TEST_DATABASE" \
    --output "$TEST_OUTPUT" \
    --results-dir "$RESULTS_DIR/full" \
    --label "R2P-GEN (full)" \
    --device cuda

echo ""
echo "=========================================================="
echo "Results in: $RESULTS_DIR/full/metrics_table.txt"
echo "Job finished at $(date)"
echo "=========================================================="