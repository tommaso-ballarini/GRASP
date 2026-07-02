#!/bin/bash
#SBATCH --job-name=r2p_single
#SBATCH --partition=edu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/single_%j.out
#SBATCH --error=logs/single_%j.err
#SBATCH --nodelist=edu02

# === SETUP ===
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

# Pulisce l'ambiente
module purge

# Carica i moduli corretti (Python 3.11 + CUDA 12.1)
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Attiva il venv LOCALE
cd ~/R2P-GEN
source venv/bin/activate

# Set environment
export R2P_CLUSTER_MODE=true
export CUDA_VISIBLE_DEVICES=0

# Create output dirs
mkdir -p logs output/single

# === FIND TEST IMAGE ===
# Cerca un'immagine. Se non la trova, usa un placeholder o esce.
TEST_IMAGE=$(find ~/data/perva-data/test/bag/alx -type f -name "*.jpg" -o -name "*.png" | head -1)

if [ -z "$TEST_IMAGE" ]; then
    echo "❌ ERRORE: Nessuna immagine trovata in ~/data/perva-data/test/bag/alx"
    # Scommenta la riga sotto se vuoi testare con un'immagine specifica fissa:
    # TEST_IMAGE="example_database/bo.png" 
    exit 1
fi

echo "Test image: $TEST_IMAGE"

# === RUN SINGLE IMAGE TEST ===
python -u full_loop.py \
    --mode single \
    --image "$TEST_IMAGE" \
    --output output/single \
    --no-refinement

echo "Job finished at $(date)"