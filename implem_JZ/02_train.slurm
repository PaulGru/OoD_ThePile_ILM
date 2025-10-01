#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=60
#SBATCH --time=20:00:00
#SBATCH --account=gxh@a100

set -euo pipefail

: "${WORK:=${HOME}}"
: "${SCRATCH:=$WORK}"

module purge >/dev/null 2>&1 || true
module load arch/a100
module load miniforge/24.9.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ilm

# Offline + caches
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export WANDB_PROJECT=${WANDB_PROJECT:-ILM-Pile}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}
export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

mkdir -p logs

CODE_DIR=${CODE_DIR:-$WORK/ThePile_OOD_ILM/OOD}

MODEL_DIR=${MODEL_DIR:-$WORK/models/distilbert-base-uncased}
TOK_DIR=${TOK_DIR:-$SCRATCH/pile_tok_ilm}
OUT_DIR=${OUT_DIR:-$PWD}
SEED=${SEED:-0}
LR=${LR:-1e-5}

MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)}
MASTER_PORT=${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 20000))}
export MASTER_ADDR MASTER_PORT

echo "[DBG] OUT_DIR=$OUT_DIR"
echo "[DBG] TOK_DIR=$TOK_DIR"
ls -ld "$TOK_DIR" || { echo "[ERR] tok dir introuvable"; exit 2; }

srun python -m torch.distributed.run \
  --standalone --nproc_per_node=1 \
  "$CODE_DIR/run_invariant_mlm.py" \
  --model_name_or_path "$MODEL_DIR" \
  --tokenized_dir "$TOK_DIR" \
  --do_train \
  --output_dir "$OUT_DIR" --overwrite_output_dir \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 6 \
  --learning_rate "$LR" \
  --nb_steps 50000 \
  --save_milestones 2500,5000,25000,50000 \
  --nb_steps_model_saving 0 \
  --report_to wandb \
  --logging_strategy steps --logging_steps 50 \
  --dataloader_num_workers 6 \
  --fp16 --half_precision_backend cuda_amp \
  --seed "$SEED" \
  --run_name "train-lr${LR}_s${SEED}" \
  --mode ilm