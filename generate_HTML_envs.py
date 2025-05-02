import os
from datasets import load_dataset, concatenate_datasets

# Charger le dataset HTMLDocumentPipeline
dataset = load_dataset("yyupenn/HTMLDocumentPipeline_form_claude_2", split="train")
print(f"Nombre total d'exemples : {len(dataset)}")  # Pour vérifier

# Fonction pour extraire "data" (HTML brut) et "code" (HTML transformé)
def extract_html_raw(example):
    return {"text": example["data"]}

def extract_html_clean(example):
    return {"text": example["code"]}

# Créer les datasets pour les deux environnements
html_raw = dataset.map(extract_html_raw, remove_columns=dataset.column_names)
html_clean = dataset.map(extract_html_clean, remove_columns=dataset.column_names)

# Split 90% / 10% pour train/validation
html_raw_split = html_raw.train_test_split(test_size=0.2, seed=42)
html_clean_split = html_clean.train_test_split(test_size=0.2, seed=42)

# Créer les dossiers de sortie
output_folder = "HTML_envs"
train_folder = os.path.join(output_folder, "train_env")
val_folder = os.path.join(output_folder, "val_env")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Fonction utilitaire pour écrire un dataset dans un fichier texte
def write_dataset_to_file(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example.get("text", "").replace("\n", " ")
            f.write(text.strip() + "\n")

# Sauvegarder les fichiers de train (séparément)
write_dataset_to_file(html_raw_split["train"], os.path.join(train_folder, "html_raw.txt"))
write_dataset_to_file(html_clean_split["train"], os.path.join(train_folder, "html_clean.txt"))

# Sauvegarder les fichiers de validation
write_dataset_to_file(html_raw_split["test"], os.path.join(val_folder, "val_ood.txt")) # val_html_raw
write_dataset_to_file(html_clean_split["test"], os.path.join(val_folder, "val_ind.txt")) # html_clean

# Créer aussi un fichier d'entraînement combiné pour eLM
all_train_examples = concatenate_datasets([html_raw_split["train"], html_clean_split["train"]])
write_dataset_to_file(all_train_examples, os.path.join(train_folder, "all_train.txt"))


print("✅ Données enregistrées dans 'HTML_envs/train_env' et 'HTML_envs/val_env'.")
print(f"- {len(html_raw_split['train'])} exemples d'entraînement pour 'html_raw'.")
print(f"- {len(html_clean_split['train'])} exemples d'entraînement pour 'html_clean'.")
print(f"- {len(html_raw_split['test'])} exemples de validation pour 'html_raw'.")
print(f"- {len(html_clean_split['test'])} exemples de validation pour 'html_clean'.")
print(f"- {len(all_train_examples)} exemples pour 'all_train' (train eLM).")
