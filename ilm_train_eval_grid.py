import os
import json
import math
import subprocess
import time
from glob import glob
from itertools import product
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force l'utilisation du premier GPU

# ------------------ CONFIG ------------------
learning_rates = [1e-6, 1e-5, 5e-5]
seeds = [0, 1, 2]
nb_steps = [250, 500, 1000, 2500]

base_dir = "runs_ilm" # runs_elm
train_file = "data/train_env" #/train_shuffled.txt
val_file = "data/val_test"


# ------------------ TRAINING ------------------
def launch_training():
    os.makedirs(base_dir, exist_ok=True)
    for lr, seed, step in product(learning_rates, seeds, nb_steps):
        exp_name = f"ilm_lr{lr}_seed{seed}_steps{step}" #elm
        out_dir = os.path.join(base_dir, exp_name)
        os.makedirs(out_dir, exist_ok=True)

        # "python3", "-m", "torch.distributed.run",
        # "--nproc_per_node=2", "run_invariant_mlm.py",
        cmd = [
            "python3", "run_invariant_mlm.py",
            "--model_name_or_path", "distilbert-base-uncased",
            "--model_type", "invariant-distilbert", #distilbert
            "--tokenizer_name", "distilbert-base-uncased",
            "--train_file", train_file,
            "--validation_file", val_file,
            "--do_train", "--do_eval",
            "--output_dir", out_dir,
            "--overwrite_output_dir",
            "--mode", "iLM", #eLM
            "--per_device_train_batch_size", "4",
            "--gradient_accumulation_steps", "2",
            "--preprocessing_num_workers", "16",
            "--learning_rate", str(lr),
            "--nb_steps", str(step),
            "--fp16",
            "--seed", str(seed)
        ]

        print(f"[TRAIN] Launching: {exp_name}")
        subprocess.run(cmd)
        print(f"[TRAIN] Finished: {exp_name}\n")

# ------------------ EVALUATION ------------------
def collect_final_metrics():
    records = []
    experiments = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    for exp in experiments:
        exp_path = os.path.join(base_dir, exp)
        metrics_path = os.path.join(exp_path, "eval_results.json")

        if not os.path.exists(metrics_path):
            print(f"[WARN] Missing eval_results.json in {exp_path}, skipping...")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        try:
            loss = metrics.get("eval_loss", float("inf"))
            perplexity = math.exp(loss) if loss < float("inf") else None
        except Exception:
            loss = None
            perplexity = None

        records.append({
            "experiment": exp,
            "eval_loss": loss,
            "perplexity": perplexity
        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(base_dir, "summary_results.csv"), index=False)
    print("[SUMMARY] Résultats enregistrés dans summary_results.csv")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    t0 = time.time()
    launch_training()
    collect_final_metrics()
    print(f"[DONE] Temps total : {round(time.time() - t0, 2)}s")
