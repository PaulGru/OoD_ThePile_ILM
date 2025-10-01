#!/bin/bash
#SBATCH --job-name=erm-train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=60
#SBATCH --time=20:00:00
#SBATCH --account=gxh@a100
#SBATCH --hint=nomultithread

set -euo pipefail

# Défauts robustes (évite SCRATCH unbound)
: "${WORK:=${HOME}}"
: "${SCRATCH:=$WORK}"
: "${SEED:=0}"
: "${LR:=1e-5}"
: "${RUN_NAME:=erm-distilbert-lr${LR}-seed${SEED}}"

module purge >/dev/null 2>&1 || true
module load arch/a100
module load miniforge/24.9.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ilm

# Offline + caches
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export HF_HOME=${HF_HOME:-$SCRATCH/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}
export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
export HF_DATASETS_IN_MEMORY=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

# Dossiers
TOK_DIR=${TOK_DIR:-$SCRATCH/pile_tok_ilm}
MODEL_DIR=${MODEL_DIR:-$WORK/models/distilbert-base-uncased}
OUT_DIR=${OUT_DIR:-$WORK/results/erm_distilbert/lr_${LR}/seed_${SEED}}
mkdir -p "$OUT_DIR"

# Sanity checks
echo "[DBG] TOK_DIR=$TOK_DIR"
ls -ld "$TOK_DIR/erm" || { echo "[ERR] '$TOK_DIR/erm' introuvable. (As-tu bien concaténé les envs ? )"; exit 2; }

# DDP standalone: port unique
MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)}
MASTER_PORT=${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 20000))}
export MASTER_ADDR MASTER_PORT
echo "[DDP] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT on $(hostname)"

# Entraînement ERM = --use_erm true (pas de --do_eval ici : on évaluera avec 03_eval.slurm)
srun python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=${SLURM_NTASKS} \
  run_invariant_mlm.py \
  --model_name_or_path "$MODEL_DIR" \
  --tokenized_dir "$TOK_DIR" \
  --do_train \
  --output_dir "$OUT_DIR" --overwrite_output_dir \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 6 \
  --learning_rate "$LR" \
  --nb_steps 50000 \
  --save_milestones 2500,5000,25000,50000 \
  --nb_steps_model_saving 0 \
  --report_to wandb --run_name "$RUN_NAME" \
  --dataloader_num_workers 6 \
  --fp16 --half_precision_backend cuda_amp \
  --mode ilm \
  --use_erm true \
  --seed "$SEED"