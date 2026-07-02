#!/bin/bash
#SBATCH --job-name=r2p_judge
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/judge/%j.out
#SBATCH --error=logs/judge/%j.err

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

mkdir -p logs/judge

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

# NOTA: il final_judge valuta immagini GIA' generate (cerca
# <concept_id>_generated.png dentro OUTPUT_DIR), quindi questa cartella deve
# essere l'output di una run verify/refine gia' completata, non una run nuova.
OUTPUT_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_debug

# ===========================================================================
# 3. Sanity check pre-run
# ===========================================================================
echo "--- Pre-run check ---"
N_GENERATED=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*_generated.png" | wc -l)
echo "Immagini *_generated.png trovate in $OUTPUT_DIR: $N_GENERATED"
if [ "$N_GENERATED" -eq 0 ]; then
    echo "❌ Nessuna immagine generata trovata in $OUTPUT_DIR."
    echo "   Il final_judge valuta output gia' prodotti: lancia prima verify_base/refine"
    echo "   su questa cartella, oppure punta OUTPUT_DIR a una run gia' completata."
    exit 1
fi

# ===========================================================================
# 4. Final Judge
# ===========================================================================
echo "Avvio final_judge su $OUTPUT_DIR..."
START=$(date +%s)

python -u flux_loop.py \
    --stage final_judge \
    --database database/database.json \
    --output $OUTPUT_DIR

END=$(date +%s)
echo "⏱️  Tempo final_judge: $((END - START))s"

# ===========================================================================
# 5. Verifica output
# ===========================================================================
echo ""
echo "--- Output check ---"
RESULTS=$OUTPUT_DIR/final_judge_results.json
if [ -f "$RESULTS" ]; then
    echo "✅ final_judge_results.json creato"
    python3 -c "
import json
with open('$RESULTS') as f:
    r = json.load(f)
print(f'   Valutati: {len(r)} concetti')
for cid, data in r.items():
    m = data.get('metrics', {})
    print(f'   {cid}: CLIP-I={m.get(\"clip_i\", 0):.3f} '
          f'| DINO-I={m.get(\"dino_i\", 0):.3f} '
          f'| TIFA={m.get(\"tifa_score\", 0):.1%}')
"
else
    echo "❌ final_judge_results.json NON trovato"
fi

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="