import os
from datasets import load_dataset, concatenate_datasets
from transformers import DistilBertTokenizerFast

# 1. Charger le dataset complet
full_dataset = load_dataset("ola13/small-the_pile", split="train")

# 2. Fonctions de prétraitement
def extract_environment(example):
    return {"environment": example["meta"].get("pile_set_name", "unknown")}

def count_tokens(example):
    token_ids = tokenizer.encode(example["text"], add_special_tokens=False)
    return {"token_count": len(token_ids)}

def compute_raw_weight(example):
    return {"raw_weight": len(example["text"].encode("utf-8"))}

# Initialiser le tokenizer et appliquer les maps
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased") # Changer le modèle si nécessaire
full_dataset = full_dataset.map(extract_environment)
full_dataset = full_dataset.map(count_tokens, batched=False)
full_dataset = full_dataset.map(compute_raw_weight, batched=False)

# 3. Listes d'environnements
train_envs = [
    "Wikipedia (en)",
    "Pile-CC",
    "EuroParl",
    "ArXiv",
    "Books3",
    "HackerNews",
    "NIH ExPorter",
    "StackExchange",
    "USPTO Backgrounds",
    "OpenSubtitles",
    "DM Mathematics",
    "FreeLaw",
    "YoutubeSubtitles",
    "OpenWebText2",
    "Github",
    "Enron Emails",
    "PubMed Central",
    "PubMed Abstracts",
]
ood_envs = [
    "Ubuntu IRC",
    "Gutenberg (PG-19)",
    "PhilPapers",
    "BookCorpus2",
]

# 4. Filtrer In-Domain et OoD
ind_dataset = full_dataset.filter(lambda x: x["environment"] in train_envs)
ood_dataset = full_dataset.filter(lambda x: x["environment"] in ood_envs)

# 5. Split In-Domain pour entraînement / validation
ind_split = ind_dataset.train_test_split(test_size=0.1, seed=42)
ind_train = ind_split["train"]
ind_val = ind_split["test"]

# 6. Création des dossiers de sortie
output_folder = "small_the_pile_env"
train_folder = os.path.join(output_folder, "train_env")
val_folder = os.path.join(output_folder, "val_env")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

def write_dataset_to_file(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(example.get("text", "").replace("\n", " ") + "\n")

# 7. Sauvegarder l'ensemble des données d'entraînement (sans distinction d'environnements)
all_train_file = os.path.join(train_folder, "all_train.txt")
write_dataset_to_file(ind_train, all_train_file)
print(f"Combined train file created with {len(ind_train)} examples.")

# 8. Sauvegarder les fichiers de validation
val_ind_file = os.path.join(val_folder, "val_ind.txt")
write_dataset_to_file(ind_val, val_ind_file)
print(f"InD validation file created with {len(ind_val)} examples.")

val_ood_file = os.path.join(val_folder, "val_ood.txt")
write_dataset_to_file(ood_dataset, val_ood_file)
print(f"OoD validation file created with {len(ood_dataset)} examples.")

print("\nEnvironnements effectivement présents dans le train set :")
found_envs = set(ind_train.unique("environment"))
print(found_envs)

missing_envs = [env for env in train_envs if env not in found_envs]
if missing_envs:
    print(f"\n Les environnements suivants sont absents du train set (aucun exemple trouvé) : {missing_envs}")

# 9. Sauvegarder les fichiers par environnement (train only)
for env in train_envs:
    subset_train = ind_train.filter(lambda x: x["environment"] == env)
    output_file = os.path.join(output_folder, f"{env}.txt")
    write_dataset_to_file(subset_train, output_file)
    print(f"Train file for '{env}' created with {len(subset_train)} examples.")

# 10. Afficher les tailles par environnement pour vérification
print("\nStatistiques par environnement (train set):")
total_train = len(ind_train)
for env in train_envs:
    count = len(ind_train.filter(lambda x: x["environment"] == env))
    print(f"- {env}: {count} exemples ({100 * count / total_train:.2f}% du total)")


all_envs = set(full_dataset.unique("environment"))
used_envs = set(train_envs + ood_envs)
ignored_envs = all_envs - used_envs

print(f"\n🌐 Environnements présents dans le dataset mais ignorés (ni InD ni OoD) : {ignored_envs}")
