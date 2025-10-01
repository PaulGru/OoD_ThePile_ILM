#!/bin/bash
#SBATCH --job-name=irm-train
#SBATCH --output=$WORK/logs/irm_train/%x_%A.out
#SBATCH --error=$WORK/logs/irm_train/%x_%A.err
#SBATCH --constraint=h100
#SBATCH --account=gxh@h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

set -euo pipefail

: "${WORK:=${HOME}}"
: "${SCRATCH:=$WORK}"

# === Paramètres à passer via --export=ALL,LR=...,SEED=... ===
LR=${LR:?LR manquant (ex: 1e-5)}
SEED=${SEED:?SEED manquante (ex: 0)}

# Répertoires
MODEL_DIR=${MODEL_DIR:-$WORK/models/distilbert-base-uncased}
TOK_DIR=${TOK_DIR:-$SCRATCH/pile_tok_ilm}   # contient les sous-dossiers des environnements + ind/ood-validation (+ eventuel 'erm')
OUT_ROOT=${OUT_ROOT:-$WORK/results/irm_distilbert}
RUN_DIR="$OUT_ROOT/lr_${LR}/seed_${SEED}"

mkdir -p "$RUN_DIR" "$WORK/logs/irm_train"

# === Modules / conda ===
module purge >/dev/null 2>&1 || true
module load arch/a100
module load miniforge/24.9.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ilm

# === Offline & caches ===
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export HF_HOME=${HF_HOME:-$SCRATCH/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}
export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

echo "[TRAIN IRM] lr=$LR seed=$SEED"
echo "[TRAIN IRM] TOK_DIR=$TOK_DIR"
echo "[TRAIN IRM] RUN_DIR=$RUN_DIR"

# DDP (1 proc)
srun python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  run_invariant_mlm.py \
  --model_name_or_path "$MODEL_DIR" \
  --tokenized_dir "$TOK_DIR" \
  --do_train \
  --output_dir "$RUN_DIR" --overwrite_output_dir \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 6 \
  --learning_rate "$LR" \
  --nb_steps 50000 \
  --save_milestones "2500,5000,25000,50000" \
  --nb_steps_model_saving 0 \
  --report_to wandb \
  --logging_strategy steps --logging_steps 50 \
  --dataloader_num_workers 6 \
  --fp16 --half_precision_backend cuda_amp \
  --seed "$SEED" \
  --mode ilm \
  --run_name "train-irm_distilbert-lr_${LR}-seed_${SEED}"