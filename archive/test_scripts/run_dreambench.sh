#!/bin/bash
#SBATCH --job-name=dreambench_gen
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=logs/dreambench_gen/%j.out
#SBATCH --error=logs/dreambench_gen/%j.err

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

# TRUCCO MAGICO: Diciamo a Python che la root del progetto è questa cartella
export PYTHONPATH=$PWD:$PYTHONPATH

# FIX: output su FM_Data (storage di progetto), non nella home
export R2P_DREAMBENCH_OUTPUT=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/output_dreambench
mkdir -p logs/dreambench_gen "$R2P_DREAMBENCH_OUTPUT"

# Esportazione Variabili d'ambiente essenziali
export HF_HOME=/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Parametri FLUX HTTP
export RECOVERY_FLUX_URL="http://127.0.0.1:8766"

echo "Avvio del server FLUX in background su GPU 0..."
CUDA_VISIBLE_DEVICES=0 python flux_server.py --port 8766 &
FLUX_PID=$!

echo "Attendo 60 secondi che il server FLUX carichi il modello in VRAM..."
sleep 60

echo "Avvio della generazione massiva DreamBench su GPU 1..."
echo "Output directory: $R2P_DREAMBENCH_OUTPUT"
CUDA_VISIBLE_DEVICES=1 python pipeline/generate_dreambench.py \
    --database database/database_db.json \
    --output "$R2P_DREAMBENCH_OUTPUT" \
    --images-per-prompt 4 \
    --batch-size 4

echo "Termino il server FLUX (PID: $FLUX_PID)..."
kill $FLUX_PID

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="