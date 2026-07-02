#!/bin/bash
#SBATCH --job-name=dreambench_db
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/dreambench_db/%j.out
#SBATCH --error=logs/dreambench_db/%j.err

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

mkdir -p logs/dreambench_db database

# ===========================================================================
# 2. Env var
# ===========================================================================
# ⚠️ PUNTA ALLA NUOVA CARTELLA DREAMBENCH CREATA NELLA FASE 1
export R2P_DATA_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/dreambench-data

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
echo "dreambench-data exists: $(ls $R2P_PERVA_DATA/test/ 2>/dev/null | head -3)"
echo "---------------------"

# ===========================================================================
# 4. Build database per DreamBench
# ===========================================================================
echo "Avvio estrazione fingerprint DreamBench (Two-Stage Pipeline)..."
# Usiamo --split test perché lo script prepare_dreambench.py ha creato la struttura "test"
python -u pipeline/build_database_db.py \
    --data-dir $R2P_DATA_DIR \
    --split test \
    --debug \
    --debug-limit 3

# NOTA: Quando vorrai processare tutti i 30 soggetti, 
# ti basterà rimuovere i flag --debug e --debug-limit 3 qui sopra.

# ===========================================================================
# 5. Verifica output
# ===========================================================================
echo ""
echo "--- Output check ---"
if [ -f database/database_perva_test.json ]; then
    echo "✅ Database creato correttamente!"
    python -c "
import json
# Il db prende il nome dallo split test a meno di config override
db_path = 'database/database.json'
try:
    with open(db_path) as f:
        db = json.load(f)
except FileNotFoundError:
    db_path = 'database/database_perva_test.json'
    with open(db_path) as f:
        db = json.load(f)

concepts = db.get('concept_dict', {})
print(f'   Concetti estratti: {len(concepts)}')
for cid, entry in list(concepts.items())[:3]:
    print(f'\n   [{cid}]:')
    print(f'     Tipo classificato: {entry.get(\"info\", {}).get(\"_entity_type\", \"MANCANTE\")}')
    print(f'     General: {entry.get(\"info\", {}).get(\"general\", \"\")}')
    print(f'     Chiavi estratte: {list(entry.get(\"info\", {}).keys())}')
"
else
    echo "❌ file del database NON trovato. Controlla i log di errore."
fi

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="