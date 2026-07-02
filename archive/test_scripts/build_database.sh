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
#SBATCH --output=logs/build_db/%j.out
#SBATCH --error=logs/build_db/%j.err

# ===========================================================================
# 1. Setup
# ===========================================================================
echo "=========================================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi | head -20
echo "=========================================================="

module purge
#module load profile/deeplrn
module load cuda/12.2
module load cudnn

cd /leonardo/home/userexternal/tballari/R2P-GEN
source $HOME/miniconda3/bin/activate FM_env

mkdir -p logs database

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

# ===========================================================================
# 3. Sanity check prima di girare
# ===========================================================================
echo "--- Sanity checks ---"
python -c "from config import Config; Config.print_summary()"
echo "Qwen3-VL exists: $(ls $R2P_MODELS_BASE/Qwen3-VL-8B-Instruct/config.json 2>/dev/null && echo YES || echo NO)"
echo "perva-data exists: $(ls $R2P_PERVA_DATA/train/ 2>/dev/null | head -3)"
echo "---------------------"

# ===========================================================================
# 4. Build database (DEBUG_MODE=True, DEBUG_LIMIT=5 da config.py)
# ===========================================================================
echo "Avvio build database..."
CONDA_PYTHON=/leonardo_work/IscrC_MUSE/tballari/envs/FM_env/bin/python
$CONDA_PYTHON -u pipeline/build_database.py \
    --split train \
    --output database/database_textdriven.json


# ===========================================================================
# 5. Verifica output
# ===========================================================================
echo ""
echo "--- Output check ---"
if [ -f database/database_textdriven.json ]; then
    echo "✅ database/database_textdriven.json creato"
    python -c "
import json
with open('database/database_textdriven.json') as f:
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
    echo "❌ database/database_textdriven.json NON trovato"
fi

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="
