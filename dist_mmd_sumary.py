import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement de la matrice MMD à partir du CSV si disponible
output_dir = "embedding_projections"
mmd_csv_path = os.path.join(output_dir, "mmd_matrix_all_envs.csv")

if os.path.exists(mmd_csv_path):
    df_mmd = pd.read_csv(mmd_csv_path, index_col=0)
else:
    raise FileNotFoundError("Le fichier 'mmd_matrix_all_envs.csv' est introuvable. Veuillez l’exporter depuis le script principal.")

# Calcul de la moyenne des distances MMD pour chaque environnement
mean_mmd = df_mmd.mean(axis=1).sort_values(ascending=False)

# Créer un tableau avec les distances moyennes
df_mean_mmd = mean_mmd.reset_index()
df_mean_mmd.columns = ['Environnement', 'Distance_MMD_moyenne']

# Sauvegarder le classement
mean_mmd_path = os.path.join(output_dir, "mean_mmd_ranking.csv")
df_mean_mmd.to_csv(mean_mmd_path, index=False)

# Affichage sous forme de barplot
plt.figure(figsize=(12, 6))
sns.barplot(data=df_mean_mmd, x="Distance_MMD_moyenne", y="Environnement", color="steelblue")
plt.title("Classement des environnements par dissimilarité moyenne (MMD)")
plt.xlabel("Distance MMD moyenne")
plt.ylabel("Environnement")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mean_mmd_ranking_plot.png"))
plt.close()
