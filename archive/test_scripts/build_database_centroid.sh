#!/bin/bash
#SBATCH --job-name=r2p_build_db
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/build_centroid/%j.out
#SBATCH --error=logs/build_centroid/%j.err

# ===========================================================================
# 1. Setup
# ===========================================================================
echo "=========================================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi | head -20
echo "=========================================================="

module purge
# Carichiamo solo CUDA, evitiamo profile/deeplrn per non corrompere Conda
module load cuda/12.2
module load cudnn

cd /leonardo/home/userexternal/tballari/R2P-GEN
# Usiamo il path assoluto di python del tuo ambiente Conda
CONDA_PYTHON=/leonardo_work/IscrC_MUSE/tballari/envs/FM_env/bin/python

mkdir -p logs/build_db database

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

# Variabile per il nome del database che ci aspettiamo
DB_FILE="database/database_centroid.json"

# ===========================================================================
# 3. Sanity check prima di girare
# ===========================================================================
echo "--- Sanity checks ---"
$CONDA_PYTHON -c "from config import Config; Config.print_summary()"
echo "Qwen3-VL exists: $(ls $R2P_MODELS_BASE/Qwen3-VL-8B-Instruct/config.json 2>/dev/null && echo YES || echo NO)"
echo "perva-data exists: $(ls $R2P_PERVA_DATA/train/ 2>/dev/null | head -3)"
echo "---------------------"

# ===========================================================================
# 4. Build database (DEBUG_MODE=True, DEBUG_LIMIT=5 da config.py)
# ===========================================================================
echo "Avvio build database..."
# Controlla che il file si chiami davvero build_database_centroid.py!
$CONDA_PYTHON -u pipeline/build_database_centroid.py \
    --split train \
    

# ===========================================================================
# 5. Verifica output
# ===========================================================================
echo ""
echo "--- Output check ---"

# Se il file centroid non c'è, controlliamo se ha usato il nome di fallback con lo split
if [ ! -f "$DB_FILE" ] && [ -f "database/database_centroid_perva_train.json" ]; then
    DB_FILE="database/database_centroid_perva_train.json"
fi

if [ -f "$DB_FILE" ]; then
    echo "✅ $DB_FILE creato"
    $CONDA_PYTHON -c "
import json
with open('$DB_FILE') as f:
    db = json.load(f)
concepts = db.get('concept_dict', {})
print(f'   Concetti: {len(concepts)}')
for cid, entry in list(concepts.items())[:2]:
    print(f'   {cid}:')
    print(f'     representative_image: {entry.get(\"representative_image\", \"MANCANTE\")}')
    print(f'     images count: {len(entry.get(\"image\", []))}')
    print(f'     info keys: {list(entry.get(\"info\", {}).keys())}')
"
else
    echo "❌ Nessun file database_centroid trovato in database/"
fi

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="