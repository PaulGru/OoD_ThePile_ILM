
import os
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

# 1. Charger l'intégralité du dataset AG News ("original" config, split "complete")
full_dataset = load_dataset("contemmcm/ag_news", "original", split="complete")

label_names = full_dataset.features["label"].names

def extract_environment(example):
    return {"environment": label_names[example["label"]]}

def count_tokens(example):
    token_ids = tokenizer.encode(example["text"], add_special_tokens=False)
    return {"token_count": len(token_ids)}

def compute_raw_weight(example):
    return {"raw_weight": len(example["text"].encode("utf-8"))}

def sanitize_filename(name):
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")

# Initialiser le tokenizer et appliquer les maps
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
full_dataset = full_dataset.map(extract_environment)
full_dataset = full_dataset.map(count_tokens, batched=False)
full_dataset = full_dataset.map(compute_raw_weight, batched=False)

# 3. Définir vos environnements InD et OoD à partir des 4 catégories
# (ajustez ces listes selon votre expérience)
train_envs = ["Top Stories", "Italia", "Top News", "Europe", "U.S.", "World", "Sports", "Health", "Software and Developement", "Sci/Tech", "Business", "Entertainment"]
ood_envs   = []

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

# 9. Sauvegarder les fichiers par environnement (train only)
for env in train_envs:
    subset_train = ind_train.filter(lambda x: x["environment"] == env)
    output_file = os.path.join(output_folder, f"{sanitize_filename(env)}.txt")
    write_dataset_to_file(subset_train, output_file)
    print(f"Train file for '{env}' created with {len(subset_train)} examples.")

# 10. Afficher les tailles par environnement pour vérification
print("\nStatistiques par environnement (train set):")
total_train = len(ind_train)
for env in train_envs:
    count = len(ind_train.filter(lambda x: x["environment"] == env))
    print(f"- {env}: {count} exemples ({100 * count / total_train:.2f}% du total)")
