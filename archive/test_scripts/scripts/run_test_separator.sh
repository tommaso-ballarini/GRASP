#!/bin/bash
#SBATCH --job-name=test_separator
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_separator/%j.out
#SBATCH --error=logs/test_separator/%j.err

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

CONDA_PYTHON=/leonardo_work/IscrC_MUSE/tballari/envs/FM_env/bin/python

source $HOME/miniconda3/bin/activate FM_env

export PYTHONPATH=$PWD
export HF_HOME=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export RECOVERY_FLUX_URL="http://127.0.0.1:8766"

OUT_DIR="/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_separator"
echo "Cleaning output folder: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo ""
echo "Flux server starting on GPU0 in background..."
CUDA_VISIBLE_DEVICES=0 $CONDA_PYTHON flux_server.py --port 8766 &
FLUX_PID=$!

echo "Waiting for the FLUX server to be ready on port 8766..."
FLUX_READY=0
for i in $(seq 1 120); do
    if curl -s -f -o /dev/null "http://127.0.0.1:8766/health"; then
        echo "✅ FLUX ready after $((i * 5))s."
        FLUX_READY=1
        break
    fi
    sleep 5
done

if [ "$FLUX_READY" -eq 0 ]; then
    echo "❌ FLUX not available. Abort."
    kill $FLUX_PID 2>/dev/null
    exit 1
fi

echo ""
echo "Running test_prompt_separator.py..."
$CONDA_PYTHON -u pipeline/test_prompt_separator.py

echo ""
echo "Test completed. Finishing the FLUX server (PID: $FLUX_PID)..."
kill $FLUX_PID

echo "=========================================================="
echo "Job finished at $(date)"
echo "Output images and JSON saved in: $OUT_DIR"
echo "=========================================================="