#!/bin/bash
#SBATCH --job-name=r2p_download
#SBATCH --partition=edu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --gres=gpu:1

# === SETUP ===
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

# Pulisce tutto per sicurezza
module purge

# --- CORREZIONE: Carichiamo ESATTAMENTE le versioni usate per installare ---
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Vai nella cartella del progetto
cd ~/R2P-GEN

# --- CORREZIONE: Attiviamo il venv locale che abbiamo appena creato ---
source venv/bin/activate

# Create logs dir se non esiste
mkdir -p logs

# === RUN ===
echo "Starting download script..."
python -u scripts/download_models.py

echo "Job finished at $(date)"