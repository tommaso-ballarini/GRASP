#!/bin/bash
#SBATCH --job-name=r2p_verify
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify/%j.out
#SBATCH --error=logs/verify/%j.err

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

mkdir -p logs

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
export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_debug

# ===========================================================================
# 3. Verify
# ===========================================================================
echo "Avvio verify_base su 5 concetti..."
START=$(date +%s)

python -u flux_loop.py \
    --stage verify_base \
    --database database/database.json \
    --output $OUTPUT_DIR

END=$(date +%s)
echo "⏱️  Time verify: $((END - START))s"

# ===========================================================================
# 4. Verifying output
# ===========================================================================
echo ""
echo "--- Output check ---"
REJECTED=$OUTPUT_DIR/rejected_concepts.json
if [ -f "$REJECTED" ]; then
    echo "✅ rejected_concepts.json creato"
    python3 -c "
import json
with open('$REJECTED') as f:
    r = json.load(f)
print(f'   Rejected: {len(r)} concetti')
for cid, data in r.items():
    print(f'   {cid}: score={data.get(\"score\", \"?\"):.3f} | method={data.get(\"details\", {}).get(\"method\", \"?\")}')
"
else
    echo "❌ rejected_concepts.json not found"
fi

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="
