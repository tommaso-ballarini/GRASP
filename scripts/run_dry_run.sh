#!/bin/bash
#SBATCH --account=IscrC_MUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=R2P_DryRun_Qwen
#SBATCH --output=logs/pipeline_30/dry_run_%j.out
#SBATCH --error=logs/pipeline_30/dry_run_%j.err

# ===========================================================================
# 1. Moduli HPC e Conda
# ===========================================================================
module purge
module load profile/deeplrn
module load cuda/12.2
module load cudnn

cd /leonardo/home/userexternal/tballari/R2P-GEN
source $HOME/miniconda3/bin/activate FM_env

export PYTHONPATH=$PWD:$PYTHONPATH

# ===========================================================================
# 2. Variabili d'Ambiente (Offline Mode)
# ===========================================================================
export HF_HOME="/leonardo_work/IscrC_MUSE/tballari/models_cache/.hf_cache"
export R2P_MODELS_BASE="/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=========================================================="
echo "🚀 INIZIO DRY RUN RECOVERY (Modello Testuale: Qwen3-VL)"
echo "=========================================================="

# ===========================================================================
# 3. Lancio test Python
# ===========================================================================
CUDA_VISIBLE_DEVICES=0 python test/test_recovery_dry_run.py

echo "=========================================================="
echo "✅ Script Dry Run terminato."
echo "=========================================================="