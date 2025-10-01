#!/bin/bash
#SBATCH --job-name=irm-eval
#SBATCH --output=$WORK/logs/irm_eval/%x_%A_%a.out
#SBATCH --error=$WORK/logs/irm_eval/%x_%A_%a.err
#SBATCH --constraint=h100
#SBATCH --account=gxh@h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3

set -euo pipefail

: "${WORK:=${HOME}}"
: "${SCRATCH:=$WORK}"

# Paramètres à passer
LR=${LR:?LR manquant (ex: 1e-5)}
SEED=${SEED:?SEED manquante (ex: 0)}
EVAL_FRACTION=${EVAL_FRACTION:-0.10}

ROOT="$WORK/results/irm_distilbert/lr_${LR}/seed_${SEED}"
TOK_DIR=${TOK_DIR:-$SCRATCH/pile_tok_ilm}

module purge >/dev/null 2>&1 || true
module load arch/a100
module load miniforge/24.9.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ilm

export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export HF_HOME=${HF_HOME:-$SCRATCH/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}
export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$WORK/logs/irm_eval"

test -d "$ROOT" || { echo "[ERR] Run introuvable: $ROOT"; exit 2; }
ls -ld "$TOK_DIR" >/dev/null || { echo "[ERR] TOK_DIR introuvable: $TOK_DIR"; exit 5; }

# Liste des ckpts (détectée dynamiquement)
mapfile -t CHECKPOINTS < <(ls -1d "$ROOT"/model-* 2>/dev/null | awk -F/ '{print $NF}' | sort -t- -k2,2n || true)
if ((${#CHECKPOINTS[@]}==0)); then
  CHECKPOINTS=(model-2500 model-5000 model-25000 model-50000)
fi

IDX=${SLURM_ARRAY_TASK_ID}
(( IDX >= 0 && IDX < ${#CHECKPOINTS[@]} )) || { echo "[ERR] SLURM_ARRAY_TASK_ID=$IDX hors bornes"; printf 'CKPTS:\n'; printf ' - %s\n' "${CHECKPOINTS[@]}"; exit 3; }

CKPT="${CHECKPOINTS[$IDX]}"
MODEL_PATH="$ROOT/$CKPT"
test -d "$MODEL_PATH" || { echo "[ERR] Checkpoint introuvable: $MODEL_PATH"; exit 4; }

OUT_EVAL="$ROOT/eval/$CKPT"
mkdir -p "$OUT_EVAL"

echo "[EVAL IRM] lr=$LR seed=$SEED ckpt=$CKPT fraction=$EVAL_FRACTION"
echo "[EVAL IRM] ROOT=$ROOT"
echo "[EVAL IRM] OUT_EVAL=$OUT_EVAL"

srun python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  run_invariant_mlm.py \
  --model_name_or_path "$MODEL_PATH" \
  --tokenized_dir "$TOK_DIR" \
  --do_eval \
  --per_device_eval_batch_size 64 \
  --seed "$SEED" \
  --eval_fraction "$EVAL_FRACTION" \
  --eval_seed "$SEED" \
  --report_to wandb \
  --output_dir "$OUT_EVAL" \
  --run_name "eval-irm_${LR}-seed${SEED}-${CKPT}"