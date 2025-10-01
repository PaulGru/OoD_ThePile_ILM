#!/bin/bash
#SBATCH --job-name=eirm-train
#SBATCH --output=$WORK/logs/eirm_train/%x_%A.out
#SBATCH --error=$WORK/logs/eirm_train/%x_%A.err
#SBATCH --constraint=h100
#SBATCH --account=gxh@h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

set -euo pipefail
: "${WORK:=${HOME}}"
: "${SCRATCH:=${WORK}}"

# ============== Paramètres passés par --export (avec défauts) ==============
LR="${LR:-5e-5}"
SEED="${SEED:-0}"

# ===== Répertoires (identique à ton style) =====
MODEL_DIR=${MODEL_DIR:-$WORK/models/distilbert-base-uncased}
TOK_DIR=${TOK_DIR:-$SCRATCH/pile_tok_ilm}  # contient les sous-dossiers des environnements + ind/ood-validation (+ éventuel 'erm')
OUT_ROOT=${OUT_ROOT:-$WORK/results/irm_distilbert}
RUN_DIR="$OUT_ROOT/eirm/lr_${LR}/seed_${SEED}"   # même pattern que IRM, juste sous-dossier 'eirm' pour éviter d'écraser

mkdir -p "$RUN_DIR"

# === Offline & caches ===
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export HF_HOME=${HF_HOME:-$SCRATCH/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}
export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

echo "[TRAIN EIRM] lr=$LR seed=$SEED"
echo "[TRAIN EIRM] MODEL_DIR=$MODEL_DIR"
echo "[TRAIN EIRM] TOK_DIR=$TOK_DIR"
echo "[TRAIN EIRM] RUN_DIR=$RUN_DIR"

# ====== Flags EIRM (ajout minimal) ======
EIRM_MODE="${EIRM_MODE:-V-IRM}"   # 'F-IRM' ou 'V-IRM'
PHI_EVERY="${PHI_EVERY:-10}"      # nb de rounds entre 2 updates de ϕ
PHI_BATCHES="${PHI_BATCHES:-2}"   # nb de mini-steps ϕ à chaque update
FREEZE_PHI="${FREEZE_PHI:-true}"  # true/false

# ====== Hyperparams (garde tes valeurs habituelles) ======
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-32}"
GRAD_ACC="${GRAD_ACC:-6}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-6}"
NB_STEPS="${NB_STEPS:-50000}"
SAVE_MILESTONES="${SAVE_MILESTONES:-2500,5000,25000,50000}"

# Activer ton environnement si besoin
module purge >/dev/null 2>&1 || true
module load arch/a100
module load miniforge/24.9.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ilm

# ====== Lancement : identique à IRM + flags EIRM en fin ======
srun python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  run_invariant_mlm.py \
  --model_name_or_path "$MODEL_DIR" \
  --tokenized_dir "$TOK_DIR" \
  --do_train \
  --output_dir "$RUN_DIR" --overwrite_output_dir \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BS" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --learning_rate "$LR" \
  --nb_steps "$NB_STEPS" \
  --save_milestones "$SAVE_MILESTONES" \
  --nb_steps_model_saving 0 \
  --report_to wandb \
  --logging_strategy steps --logging_steps 50 \
  --dataloader_num_workers "$DATALOADER_WORKERS" \
  --fp16 --half_precision_backend cuda_amp \
  --seed "$SEED" \
  --mode ilm \
  --run_name "train-eirm_distilbert-lr_${LR}-seed_${SEED}" \
  --eirm_mode "$EIRM_MODE" \
  --phi_update_every "$PHI_EVERY" \
  --phi_update_batches "$PHI_BATCHES" \
  $( [[ "$FREEZE_PHI" == "true" ]] && echo --freeze_phi_at_start true )

echo "[TRAIN EIRM] done."