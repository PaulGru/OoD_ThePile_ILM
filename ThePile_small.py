import os
import random
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

# Charger le dataset complet (seul split disponible) et le découper en 80/20
full_dataset = load_dataset("ola13/small-the_pile", split="train")
split_datasets = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets["train"]  # Pour l'entraînement
test_dataset = split_datasets["test"]    # Pour l'évaluation (validation InD et OoD)

print(f"Nombre d'exemples train: {len(train_dataset)}")
print(f"Nombre d'exemples test: {len(test_dataset)}")

# Extraire l'environnement à partir du champ "meta"
def extract_environment(example):
    env = example["meta"].get("pile_set_name", "unknown")
    return {"environment": env}

train_dataset = train_dataset.map(extract_environment)
test_dataset = test_dataset.map(extract_environment)

# Compter le nombre de tokens via DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
def count_tokens(example):
    token_ids = tokenizer.encode(example["text"], add_special_tokens=False)
    return {"token_count": len(token_ids)}

train_dataset = train_dataset.map(count_tokens, batched=False)
test_dataset = test_dataset.map(count_tokens, batched=False)

# Récupérer la liste unique des environnements depuis le train
environments = list(set(train_dataset["environment"]))
print("Environnements trouvés dans le train :", environments)

# Définir InD et OoD à partir des environnements du train
random.shuffle(environments)
num_train_env = int(0.6 * len(environments))
train_envs = environments[:num_train_env]
ood_envs = environments[num_train_env:]
print("Environnements InD :", train_envs)
print("Environnements OoD :", ood_envs)

# Créer un dossier pour sauvegarder les fichiers
output_folder = "small_the_pile_env"
os.makedirs(output_folder, exist_ok=True)

def write_dataset_to_file(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example.get("text", "")
            f.write(text.replace("\n", " ") + "\n")

# Générer les fichiers d'entraînement par environnement InD à partir du train
for env in train_envs:
    subset = train_dataset.filter(lambda x: x["environment"] == env)
    output_file = os.path.join(output_folder, f"{env}.txt")
    write_dataset_to_file(subset, output_file)
    print(f"Fichier d'entraînement pour '{env}' créé avec {len(subset)} exemples.")

# Pour la validation, utiliser directement le split test en le filtrant :
# - Validation InD : les exemples du test appartenant aux environnements InD
# - Validation OoD : les exemples du test appartenant aux environnements OoD

val_ind = test_dataset.filter(lambda x: x["environment"] in train_envs)
val_ood = test_dataset.filter(lambda x: x["environment"] in ood_envs)

val_ind_file = os.path.join(output_folder, "val_ind.txt")
val_ood_file = os.path.join(output_folder, "val_ood.txt")
write_dataset_to_file(val_ind, val_ind_file)
write_dataset_to_file(val_ood, val_ood_file)

print(f"Validation InD créée avec {len(val_ind)} exemples.")
print(f"Validation OoD créée avec {len(val_ood)} exemples.")

# Affichage des statistiques pour le train (par environnement)
total_train_examples = len(train_dataset)
total_train_tokens = sum(train_dataset["token_count"])

print("\nStatistiques par environnement dans le train:")
for env in environments:
    env_subset = train_dataset.filter(lambda x: x["environment"] == env)
    num_examples = len(env_subset)
    total_tokens_env = sum(env_subset["token_count"])
    pct_examples = 100 * num_examples / total_train_examples
    pct_tokens = 100 * total_tokens_env / total_train_tokens
    print(f"Environnement: {env}")
    print(f"  Exemples: {num_examples} ({pct_examples:.2f}% du total)")
    print(f"  Tokens: {total_tokens_env} ({pct_tokens:.2f}% du total)")

# Création du fichier combiné pour eLM après avoir généré tous les fichiers d'environnements
all_train = train_dataset.filter(lambda x: x["environment"] in train_envs)
output_file = os.path.join(output_folder, "all_train.txt")
write_dataset_to_file(all_train, output_file)
print(f"Fichier combiné 'all_train.txt' créé avec {len(all_train)} exemples.")

