import os
import json
import math
import subprocess
import time
from glob import glob
from itertools import product
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 0 eLM, GPU 1 iLM

# ------------------ CONFIG ------------------
learning_rates = [1e-5] # [5e-5]
seeds = [2, 3]

nb_steps = 7500
save_steps = 500

base_dirs = {
    "ilm": "runs_ilm",
    "elm": "runs_elm"
}

val_file = "data/val_test/val_ind.txt"
ood_file = "data/val_test/val_ood.txt"


# ------------------ TRAINING ------------------
def launch_training(model_key):
    base_dir = base_dirs[model_key]

    if model_key == "elm":
        train_file = "data/train_erm.txt"
    
    elif model_key == "ilm":
        train_file = "data/train_env"

    os.makedirs(base_dir, exist_ok=True)
    for lr in learning_rates:
        for seed in seeds:
            exp_name = f"model_{base_dir}_lr{lr}_seed{seed}"
            out_dir = os.path.join(base_dir, exp_name)
            os.makedirs(out_dir, exist_ok=True)

            print(f"\nLancement de l'entraînement: {exp_name}")

            # "python3",
            cmd = [
                "python3", "-m", "torch.distributed.run",
                "--nproc_per_node=1",
                "--master_port", "29500", # GPU 0 : 29501, GPU 1 : 29500
                "run_invariant_mlm.py",
                "--model_name_or_path", "distilbert-base-uncased",
                "--train_file", train_file,
                "--validation_file", val_file,
                "--ood_validation_file", ood_file,
                "--do_train", "--do_eval",
                "--output_dir", out_dir,
                "--overwrite_output_dir",
                "--nb_steps_model_saving", str(save_steps),
                "--max_seq_length", "128",
                "--per_device_train_batch_size", "16",
                "--gradient_accumulation_steps", "3",
                "--preprocessing_num_workers", "16",
                "--learning_rate", str(lr),
                "--nb_steps", str(nb_steps),
                "--fp16",
                "--seed", str(seed),
                "--run_name", exp_name,
            ]

            print(f"[TRAIN] Launching: {exp_name}")
            subprocess.run(cmd)
            print(f"[TRAIN] Finished: {exp_name}\n")


if __name__ == "__main__":
    t0 = time.time()
    launch_training("elm") # "ilm"
    print(f"[DONE] Temps total : {round(time.time() - t0, 2)}s")
