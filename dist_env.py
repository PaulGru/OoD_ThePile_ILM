import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
from tqdm import tqdm

# Initialisation
env_dir = "small_the_pile_env"  # dossier contenant les fichiers par environnement
output_dir = "embedding_projections"
os.makedirs(output_dir, exist_ok=True)

nb_exemples_par_env = 200  # nombre d'exemples à échantillonner par environnement

# Charger tokenizer et modèle BERT
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.eval()
model.cuda()

# Récupérer tous les fichiers txt disponibles (même OoD)
env_files = [f for f in os.listdir(env_dir) if f.endswith(".txt") and f not in {"all_train.txt"}]

all_embeddings = []
all_labels = []

with torch.no_grad():
    for filename in env_files:
        filepath = os.path.join(env_dir, filename)
        env_name = filename.replace(".txt", "")

        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if len(line.strip()) > 10][:nb_exemples_par_env]

        print(f"\nProcessing {env_name} ({len(lines)} examples)...")

        for line in tqdm(lines):
            inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            all_embeddings.append(emb)
            all_labels.append(env_name)

# Transformation en numpy
all_embeddings = np.stack(all_embeddings)

# Réduction de dimension : PCA (2D)
pca = PCA(n_components=2)
proj_pca = pca.fit_transform(all_embeddings)

# t-SNE (plus lent mais plus lisible si non-linéaire)
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
proj_tsne = tsne.fit_transform(all_embeddings)

# Affichage
sns.set(style="whitegrid", rc={"figure.figsize": (12, 7)})
def plot_proj(proj, title, filename):
    plt.figure()
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=all_labels, palette="tab20", s=40, edgecolor=None, alpha=0.8)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Environnements")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_proj(proj_pca, "Projection PCA de tous les environnements", "pca_projection_all_envs.png")
plot_proj(proj_tsne, "Projection t-SNE de tous les environnements", "tsne_projection_all_envs.png")
print(f"Plots enregistrés dans le dossier : {output_dir}")
