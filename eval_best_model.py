import os
import csv
import subprocess

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

# Dossier contenant les différents runs (chacun avec un best_model/)
runs_dir = "runs_elm"
eval_file = "data/val_test/val_ood.txt"
results_file = "eval_elm_ood.csv"

# Liste des répertoires à traiter
run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]

# Liste pour stocker les résultats
results = []

for run in run_dirs:
    run_path = os.path.join(runs_dir, run)
    best_model_path = os.path.join(run_path, "best_model")

    if not os.path.isdir(best_model_path):
        print(f"Pas de best_model dans {run}")
        continue

    print(f"Évaluation de {run}")

    # Appel au script d'éval avec les bons paramètres
    cmd = [
        "python3", "run_invariant_mlm.py",
        "--model_name_or_path", best_model_path,
        "--tokenizer_name", "distilbert-base-uncased",
        "--validation_file", eval_file,
        "--output_dir", os.path.join(run_path, "ood_eval"),
        "--do_eval",
        "--eval_type", "ood"
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Extraction des résultats de la sortie du script
    eval_loss = None
    perplexity = None
    for line in completed.stdout.splitlines():
        if "eval_loss" in line:
            eval_loss = float(line.split("=")[-1].strip())
        if "perplexity" in line:
            perplexity = float(line.split("=")[-1].strip())

    results.append({
        "run": run,
        "eval_loss": eval_loss,
        "perplexity": perplexity
    })

# Écriture dans un CSV
with open(results_file, "w", newline="") as csvfile:
    fieldnames = ["run", "eval_loss", "perplexity"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for res in results:
        writer.writerow(res)

print(f"Résultats enregistrés dans {results_file}")
