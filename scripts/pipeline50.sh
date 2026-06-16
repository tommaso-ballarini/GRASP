#!/bin/bash
#SBATCH --job-name=r2p_pipeline_50
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/pipeline_50/%j.out
#SBATCH --error=logs/pipeline_50/%j.err

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

OUTPUT_DIR=/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_50

mkdir -p $OUTPUT_DIR

# ===========================================================================
# 3. Build database (50 concetti)
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 1/3 — BUILD DATABASE (50 concetti)"
echo "=========================================================="
START=$(date +%s)

python -u pipeline/build_dataset.py \
    --split train \
    --debug \
    --debug-limit 50

END=$(date +%s)
echo "⏱️  Build completato in $((END - START))s"

if [ ! -f database/database.json ]; then
    echo "❌ database.json non trovato — interruzione pipeline"
    exit 1
fi

python3 -c "
import json
with open('database/database.json') as f:
    db = json.load(f)
print(f'   ✅ Concetti nel database: {len(db[\"concept_dict\"])}')
"

# ===========================================================================
# 4. Generate (50 concetti)
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 2/3 — GENERATE (50 concetti)"
echo "=========================================================="
START=$(date +%s)

python -u flux_loop.py \
    --stage generate_only \
    --database database/database.json \
    --output $OUTPUT_DIR

END=$(date +%s)
echo "⏱️  Generate completato in $((END - START))s"

PNG_COUNT=$(ls $OUTPUT_DIR/*_generated.png 2>/dev/null | wc -l)
echo "   PNG generati: $PNG_COUNT"

if [ "$PNG_COUNT" -eq 0 ]; then
    echo "❌ Nessuna immagine generata — interruzione pipeline"
    exit 1
fi

# ===========================================================================
# 5. Verify (50 concetti)
# ===========================================================================
echo ""
echo "=========================================================="
echo "STAGE 3/3 — VERIFY (50 concetti)"
echo "=========================================================="
START=$(date +%s)

python -u flux_loop.py \
    --stage verify_base \
    --database database/database.json \
    --output $OUTPUT_DIR

END=$(date +%s)
echo "⏱️  Verify completato in $((END - START))s"

# ===========================================================================
# 6. Report finale
# ===========================================================================
echo ""
echo "=========================================================="
echo "REPORT FINALE"
echo "=========================================================="

python3 -c "
import json, os

with open('database/database.json') as f:
    db = json.load(f)
total = len(db['concept_dict'])

output_dir = '$OUTPUT_DIR'
rejected_path = os.path.join(output_dir, 'rejected_concepts.json')

png_count = len([f for f in os.listdir(output_dir) if f.endswith('_generated.png')])

print(f'  Concetti nel database : {total}')
print(f'  Immagini generate     : {png_count}')

if os.path.exists(rejected_path):
    with open(rejected_path) as f:
        rejected = json.load(f)
    passed = total - len(rejected)
    print(f'  Verify passed        : {passed}/{total}')
    print(f'  Verify rejected      : {len(rejected)}/{total}')
    if rejected:
        print(f'  Metodi di reject:')
        methods = {}
        for cid, data in rejected.items():
            m = data.get('details', {}).get('method', 'unknown')
            methods[m] = methods.get(m, 0) + 1
        for m, count in methods.items():
            print(f'    {m}: {count}')
else:
    print('  ⚠️  rejected_concepts.json non trovato')

if os.path.exists(os.path.join(output_dir, 'prompts.json')):
    print('  ✅ prompts.json presente')
"

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="
