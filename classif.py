# pip install pandas

import pandas as pd

def main():
    # 1. Lire le CSV issu du calcul par profession
    df = pd.read_csv("stats_genre_par_profession.csv")
    # Colonnes attendues : ['profession', 'male', 'pct_male', 'female', 'pct_female', 'total']

    # 2. Assigner la catégorie majoritaire à chaque profession
    df["category"] = df.apply(
        lambda row: "homme" if row["male"] > row["female"] else "femme",
        axis=1
    )

    # 3. Agréger par catégorie : somme des hommes et des femmes
    summary = df.groupby("category")[["male", "female"]].sum()

    # 4. Calculer total et pourcentages
    summary["total"]       = summary["male"] + summary["female"]
    summary["pct_male"]    = 100 * summary["male"]   / summary["total"]
    summary["pct_female"]  = 100 * summary["female"] / summary["total"]

    # 5. Afficher et sauvegarder
    print(summary)
    summary.to_csv("stats_genre_par_categorie.csv")

if __name__ == "__main__":
    main()
