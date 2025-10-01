#!/bin/bash
#SBATCH --job-name=ilm-eval-grid
#SBATCH --output=$WORK/logs/ilm_eval_grid/%x_%A_%a.out
#SBATCH --error=$WORK/logs/ilm_eval_grid/%x_%A_%a.err
#SBATCH --constraint=a100
#SBATCH --account=gxh@a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-39

set -euo pipefail

: "${WORK:=${HOME}}"
: "${SCRATCH:=$WORK}"
export WORK SCRATCH

module purge >/dev/null 2>&1 || true
module load arch/a100
module load miniforge/24.9.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ilm

# Offline + caches
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export HF_HOME=${HF_HOME:-$SCRATCH/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}
export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$WORK/logs/ilm_eval_grid"

# --- chemins et params ---
TOK_DIR=${TOK_DIR:-$SCRATCH/pile_tok_ilm}
MODEL_ROOT=${MODEL_ROOT:-$WORK/results/distilbert_ilm_grid}
EVAL_FRACTION=${EVAL_FRACTION:-0.1}

declare -a LRS=("1e-5" "5e-5")
declare -a SEEDS=(0 1 2 3 4)
declare -a CKPTS=("model-2500" "model-5000" "model-25000" "model-50000")
# Si tu veux aussi le "final" ajoute: "final" et règle la résolution de chemin plus bas.

n_lr=${#LRS[@]}
n_seed=${#SEEDS[@]}
n_ckpt=${#CKPTS[@]}

i=${SLURM_ARRAY_TASK_ID}
lr_idx=$(( i / (n_seed*n_ckpt) ))
rest=$(( i % (n_seed*n_ckpt) ))
seed_idx=$(( rest / n_ckpt ))
ckpt_idx=$(( rest % n_ckpt ))

LR=${LRS[$lr_idx]}
SEED=${SEEDS[$seed_idx]}
CKPT=${CKPTS[$ckpt_idx]}

OUT_DIR="$MODEL_ROOT/lr_${LR}/seed_${SEED}"
CKPT_PATH="$OUT_DIR/$CKPT"
# (option "final") : si CKPT == "final", alors CKPT_PATH="$OUT_DIR"
EVAL_OUT="$OUT_DIR/eval/$CKPT"
RUN_NAME="eval-lr_${LR}-seed_${SEED}-${CKPT}"

echo "[EVAL] lr=$LR seed=$SEED ckpt=$CKPT"
ls -ld "$OUT_DIR" || { echo "[ERR] OUT_DIR introuvable: $OUT_DIR"; exit 2; }
if [[ "$CKPT" == "final" ]]; then
  CKPT_PATH="$OUT_DIR"
fi
ls -ld "$CKPT_PATH" || { echo "[ERR] ckpt introuvable: $CKPT_PATH"; exit 2; }
ls -ld "$TOK_DIR" || { echo "[ERR] TOK_DIR introuvable: $TOK_DIR"; exit 2; }
mkdir -p "$EVAL_OUT"

# Un seul process; on garde torch.distributed.run pour rester homogène.
srun python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  "$WORK/ThePile_OOD_ILM/OOD/run_invariant_mlm.py" \
  --model_name_or_path "$CKPT_PATH" \
  --tokenized_dir "$TOK_DIR" \
  --do_eval \
  --per_device_eval_batch_size 64 \
  --seed "$SEED" \
  --eval_fraction "$EVAL_FRACTION" \
  --eval_seed "$SEED" \
  --report_to wandb \
  --output_dir "$EVAL_OUT" \
  --run_name "$RUN_NAME"