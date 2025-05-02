from datasets import load_dataset
import random
import os

# Paramètres
DATASET_NAME = "wikimedia/wikipedia"
EN_LANG = "20231101.en"
FA_LANG = "20231101.fa"
TRAIN_SIZE = 10000
VAL_SIZE = 1000
MIN_LEN = 10  # minimum de mots dans un texte
SEED = 42
SAVE_DIR = "Wiki_clean"  # Répertoire de sauvegarde

random.seed(SEED)

# Fonction pour écrire les fichiers texte
def write_txt(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex["text"].replace("\n", " ") + "\n")

# Charger les jeux de données anglais et farsi
print("Chargement des jeux de données...")
en_dataset = load_dataset(DATASET_NAME, EN_LANG, split="train[:15000]")
fa_dataset = load_dataset(DATASET_NAME, FA_LANG, split="train[:15000]")

# Nettoyer : retirer les textes trop courts
print("Nettoyage des données...")
cleaned_en = [ex for ex in en_dataset if ex["text"] and len(ex["text"].split()) > MIN_LEN]
cleaned_fa = [ex for ex in fa_dataset if ex["text"] and len(ex["text"].split()) > MIN_LEN]

# Mélanger
random.shuffle(cleaned_en)
random.shuffle(cleaned_fa)

# Prendre 10k pour train + 1k pour validation
train_en = cleaned_en[:TRAIN_SIZE]
val_en = cleaned_en[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

train_fa = cleaned_fa[:TRAIN_SIZE]
val_fa = cleaned_fa[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

# Fusionner les environnements
train = train_en + train_fa
val = val_en + val_fa

# Re-mélanger
random.shuffle(train)
random.shuffle(val)

# Sauvegarder
print("Sauvegarde des fichiers...")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "val_env"), exist_ok=True)

write_txt(train, os.path.join(SAVE_DIR, "train", "all_train.txt"))
write_txt(val, os.path.join(SAVE_DIR, "val_env", "val_ind.txt"))
write_txt(train_en, os.path.join(SAVE_DIR, "env_en.txt"))
write_txt(train_fa, os.path.join(SAVE_DIR, "env_far.txt"))

print("✅ Nettoyage terminé !")
