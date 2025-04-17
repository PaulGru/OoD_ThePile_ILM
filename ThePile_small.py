import os
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

# 1. Charger l'intégralité du dataset (pas de split initial)
full_dataset = load_dataset("ola13/small-the_pile", split="train")

# 2. Définir les fonctions de prétraitement
def extract_environment(example):
    env = example["meta"].get("pile_set_name", "unknown")
    return {"environment": env}

def count_tokens(example):
    token_ids = tokenizer.encode(example["text"], add_special_tokens=False)
    return {"token_count": len(token_ids)}

def compute_raw_weight(example):
    return {"raw_weight": len(example["text"].encode("utf-8"))}

# Extraire l'environnement pour chaque exemple
full_dataset = full_dataset.map(extract_environment)

# Initialiser le tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Calcul du nombre de tokens et du poids brut
full_dataset = full_dataset.map(count_tokens, batched=False)
full_dataset = full_dataset.map(compute_raw_weight, batched=False)

# 3. Utiliser des listes fixes pour définir les environnements InD et OoD
train_envs = [ 
    "Wikipedia (en)",
    "Pile-CC",
    "EuroParl",
    "ArXiv",
    "Books3",
]

ood_envs = [ 
    "Enron Emails",
    "OpenWebText2",
    "PubMed Abstracts",
    "Github",
    "Ubuntu IRC",
    "PubMed Central",
    "Gutenberg (PG-19)",
    "DM Mathematics",
    "HackerNews",
    "StackExchange",
    "YoutubeSubtitles",
    "FreeLaw",
    "USPTO Backgrounds",
]

print("Environnements InD :", train_envs)
print("Environnements OoD :", ood_envs)

# 4. Créer des sous-ensembles du dataset complet selon ces environnements
# Sous-ensemble InD
ind_dataset = full_dataset.filter(lambda x: x["environment"] in train_envs)
# Sous-ensemble OoD
ood_dataset = full_dataset.filter(lambda x: x["environment"] in ood_envs)

# 5. Sur le sous-ensemble InD, effectuer un split pour extraire un held-out de validation (InD)
# Par exemple, réserver 10% pour la validation InD
ind_split = ind_dataset.train_test_split(test_size=0.1, seed=42)
ind_train = ind_split["train"]
ind_val = ind_split["test"]

# 6. Créer des dossiers pour sauvegarder les fichiers
output_folder = "small_the_pile_env"
train_folder = os.path.join(output_folder, "train_env")
val_folder = os.path.join(output_folder, "val_env")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

def write_dataset_to_file(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example.get("text", "")
            # Remplacer les sauts de ligne par des espaces pour uniformiser le format
            f.write(text.replace("\n", " ") + "\n")

# Sauvegarder chaque environnement InD individuellement (optionnel)
for env in train_envs:
    subset = ind_dataset.filter(lambda x: x["environment"] == env)
    output_file = os.path.join(output_folder, f"{env}.txt")
    write_dataset_to_file(subset, output_file)
    print(f"Fichier pour environnement '{env}' créé avec {len(subset)} exemples.")

# Fichier combiné pour l'entraînement InD (après le split, données de training uniquement)
all_train_file = os.path.join(train_folder, "all_train.txt")
write_dataset_to_file(ind_train, all_train_file)
print(f"Fichier d'entraînement InD combiné créé avec {len(ind_train)} exemples.")

# Fichier de validation InD (held-out)
val_ind_file = os.path.join(val_folder, "val_ind.txt")
write_dataset_to_file(ind_val, val_ind_file)
print(f"Fichier de validation InD créé avec {len(ind_val)} exemples.")

# Fichier de validation OoD (utilisation du sous-ensemble OoD complet)
val_ood_file = os.path.join(val_folder, "val_ood.txt")
write_dataset_to_file(ood_dataset, val_ood_file)
print(f"Fichier de validation OoD créé avec {len(ood_dataset)} exemples.")

# 7. Afficher les statistiques par environnement dans le training set (all_train.txt)
total_train_examples = len(ind_train)
total_train_tokens = sum(ind_train["token_count"])

print("\nStatistiques par environnement dans le training set (all_train.txt) :")
for env in train_envs:
    env_subset = ind_train.filter(lambda x: x["environment"] == env)
    num_examples = len(env_subset)
    total_tokens_env = sum(env_subset["token_count"])
    pct_examples = 100 * num_examples / total_train_examples if total_train_examples > 0 else 0
    pct_tokens = 100 * total_tokens_env / total_train_tokens if total_train_tokens > 0 else 0
    print(f"Environnement: {env}")
    print(f"  Exemples: {num_examples} ({pct_examples:.2f}% du total)")
    print(f"  Tokens: {total_tokens_env} ({pct_tokens:.2f}% du total)")
