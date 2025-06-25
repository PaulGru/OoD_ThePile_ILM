import os
import random
from datasets import load_dataset, concatenate_datasets
from transformers import DistilBertTokenizerFast

MAX_PER_ENV = 1000

# 1. Charger le dataset complet
dataset_val = load_dataset("monology/pile-test-val", split="validation")
dataset_test = load_dataset("monology/pile-test-val", split="test")
dataset = concatenate_datasets([dataset_val, dataset_test])

# 2. Fonctions de prétraitement
def extract_environment(example):
    return {"environment": example["meta"].get("pile_set_name", "unknown")}

dataset = dataset.map(extract_environment)

# "BookCorpus2", "OpenSubtitles", "PhilPapers", "NIH ExPorter",
# 3. Listes d'environnements
train_envs = [    
    "EuroParl",
    "FreeLaw",
    "DM Mathematics",
    "YoutubeSubtitles",
    "USPTO Backgrounds",
    "ArXiv",
    "Books3",
    "Wikipedia (en)",
    "StackExchange",
    "HackerNews",
    "Pile-CC",
]

ood_envs = [
    "Github",
    "Ubuntu IRC",
    "OpenWebText2",
    "Enron Emails",
    "PubMed Central",
    "PubMed Abstracts",
    "Gutenberg (PG-19)",
]

# 4. Filtrer In-Domain et OoD
ind_dataset = dataset.filter(lambda x: x["environment"] in train_envs)
ood_dataset = dataset.filter(lambda x: x["environment"] in ood_envs)

# 5. Split In-Domain pour entraînement / validation
ind_split = ind_dataset.train_test_split(test_size=0.1, seed=0)
ind_train = ind_split["train"]
ind_val = ind_split["test"]

# 6. Création des dossiers de sortie
output_folder = "small_the_pile"
train_folder = os.path.join(output_folder, "train_env")
val_folder = os.path.join(output_folder, "val_test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

def write_dataset_to_file(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(example.get("text", "").replace("\n", " ") + "\n")

# 7. Sauvegarder l'ensemble des données d'entraînement (sans distinction d'environnements)
all_train_file = os.path.join(output_folder, "all_train.txt")
write_dataset_to_file(ind_train, all_train_file)

MAX_PER_ENV = 1000

# --- In-Domain (val_ind) équilibré "tronqué"
ind_val_limited = []
for env in train_envs:
    subset = ind_val.filter(lambda x: x["environment"] == env)
    n_samples = min(len(subset), MAX_PER_ENV)
    if n_samples > 0:
        sampled = subset.shuffle(seed=42).select(range(n_samples))
        ind_val_limited.append(sampled)

ind_val_trimmed = concatenate_datasets(ind_val_limited)
print(f"val_ind tronqué : {len(ind_val_trimmed)} exemples (max {MAX_PER_ENV}/env)")

# --- Out-of-Domain (val_ood) équilibré "tronqué"
ood_val_limited = []
for env in ood_envs:
    subset = ood_dataset.filter(lambda x: x["environment"] == env)
    n_samples = min(len(subset), MAX_PER_ENV)
    if n_samples > 0:
        sampled = subset.shuffle(seed=42).select(range(n_samples))
        ood_val_limited.append(sampled)

ood_val_trimmed = concatenate_datasets(ood_val_limited)
print(f"val_ood tronqué : {len(ood_val_trimmed)} exemples (max {MAX_PER_ENV}/env)")

# 9. Sauvegarder les fichiers de validation
val_ind_file = os.path.join(val_folder, "val_ind.txt")
write_dataset_to_file(ind_val_trimmed, val_ind_file)

val_ood_file = os.path.join(val_folder, "val_ood.txt")
write_dataset_to_file(ood_val_trimmed, val_ood_file)

# 10. Sauvegarder les fichiers par environnement (train only)
for env in train_envs:
    subset_train = ind_train.filter(lambda x: x["environment"] == env)
    output_file = os.path.join(train_folder, f"{env}.txt")
    write_dataset_to_file(subset_train, output_file)
    print(f"Train file for '{env}' created with {len(subset_train)} examples.")

# 11. Afficher les tailles par environnement pour vérification
print("\nStatistiques par environnement (train set):")
total_train = len(ind_train)
for env in train_envs:
    count = len(ind_train.filter(lambda x: x["environment"] == env))
    print(f"- {env}: {count} exemples ({100 * count / total_train:.2f}% du total)")
