import os
import random
from tqdm import tqdm
from datasets import load_dataset

# Partie 1 : Télécharger et préparer Wikitext-2 si besoin
def download_wikitext2(output_dir="Wikitext-2"):
    if not os.path.exists(output_dir):
        print("Téléchargement de Wikitext-2 via HuggingFace datasets...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        os.makedirs(output_dir, exist_ok=True)

        for split, filename in zip(["train", "validation", "test"], ["train.txt", "valid.txt", "test.txt"]):
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                for line in dataset[split]["text"]:
                    f.write(line.strip() + "\n")
        print("Téléchargement terminé.\n")
    else:
        print("Wikitext-2 déjà présent, téléchargement ignoré.\n")

# Partie 2 : Inversion des termes genrés
GENDER_PAIRS = [
    ("he", "she"), ("him", "her"), ("his", "hers"),
    ("man", "woman"), ("men", "women"),
    ("boy", "girl"), ("boys", "girls"),
    ("father", "mother"), ("fathers", "mothers"),
    ("son", "daughter"), ("sons", "daughters"),
    ("brother", "sister"), ("brothers", "sisters"),
    ("uncle", "aunt"), ("uncles", "aunts"),
    ("husband", "wife"), ("husbands", "wives"),
    ("actor", "actress"), ("actors", "actresses"),
    ("king", "queen"), ("kings", "queens"),
    ("waiter", "waitress"), ("waiters", "waitresses"),
    ("prince", "princess"), ("princes", "princesses"),
    ("mr.", "mrs."), ("mr", "mrs"),
    ("male", "female"), ("males", "females"),
    ("gentleman", "lady"), ("gentlemen", "ladies"),
    ("businessman", "businesswoman"), ("businessmen", "businesswomen"),
    ("boyfriend", "girlfriend"), ("boyfriends", "girlfriends"),
    ("stepfather", "stepmother"), ("stepfathers", "stepmothers"),
    ("spokesman", "spokeswoman"), ("spokesmen", "spokeswomen"),
    ("hero", "heroine"), ("heroes", "heroines"),
    ("grandson", "granddaughter"), ("grandsons", "granddaughters"),
]

def create_gender_dict(pairs):
    gender_dict = {}
    for male, female in pairs:
        gender_dict[male.lower()] = female.lower()
        gender_dict[female.lower()] = male.lower()
    return gender_dict

GENDER_DICT = create_gender_dict(GENDER_PAIRS)

def swap_gender_terms(sentence, gender_dict):
    tokens = sentence.split()
    swapped_tokens = []
    for token in tokens:
        lower_token = token.lower()
        if lower_token in gender_dict:
            swapped = gender_dict[lower_token]
            if token[0].isupper():
                swapped = swapped.capitalize()
            swapped_tokens.append(swapped)
        else:
            swapped_tokens.append(token)
    return ' '.join(swapped_tokens)

def prepare_wikitext2(output_dir="Wikitext-2", p_env_a=0.5):
    # Charger les données
    train_path = os.path.join(output_dir, "train.txt")
    valid_path = os.path.join(output_dir, "valid.txt")
    test_path = os.path.join(output_dir, "test.txt")

    with open(train_path, 'r', encoding='utf-8') as f:
        train_lines = [line.strip() for line in f if line.strip()]
    with open(valid_path, 'r', encoding='utf-8') as f:
        valid_lines = [line.strip() for line in f if line.strip()]
    with open(test_path, 'r', encoding='utf-8') as f:
        test_lines = [line.strip() for line in f if line.strip()]

    # Créer env_A et env_B
    env_a_lines = []
    env_b_lines = []
    for line in tqdm(train_lines, desc="Création des environnements A et B"):
        if random.random() < p_env_a:
            env_a_lines.append(line)
        else:
            swapped_line = swap_gender_terms(line, GENDER_DICT)
            env_b_lines.append(swapped_line)

    # Sauvegarder env_A.txt et env_B.txt
    with open(os.path.join(output_dir, "env_A.txt"), "w", encoding="utf-8") as f:
        for line in env_a_lines:
            f.write(line + "\n")
    with open(os.path.join(output_dir, "env_B.txt"), "w", encoding="utf-8") as f:
        for line in env_b_lines:
            f.write(line + "\n")

    # Créer train_env/ et val_env/
    train_env_dir = os.path.join(output_dir, "train_env")
    val_env_dir = os.path.join(output_dir, "val_env")
    os.makedirs(train_env_dir, exist_ok=True)
    os.makedirs(val_env_dir, exist_ok=True)

    # all_train.txt = concat env_A + env_B
    with open(os.path.join(train_env_dir, "all_train.txt"), "w", encoding="utf-8") as f:
        for line in env_a_lines + env_b_lines:
            f.write(line + "\n")

    # val_ind.txt = validation set
    with open(os.path.join(val_env_dir, "val_ind.txt"), "w", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")

    # val_ood.txt = test set
    with open(os.path.join(val_env_dir, "val_ood.txt"), "w", encoding="utf-8") as f:
        for line in test_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    random.seed(42)  # Pour que le split soit reproductible
    download_wikitext2()
    prepare_wikitext2()
    print("\n✅ Données préparées correctement dans le dossier 'Wikitext-2/' !")
