# pip install datasets pandas

from datasets import load_dataset, concatenate_datasets
import pandas as pd

def main():
    # 1. Charger le split “train”
    ds_train = load_dataset("LabHC/bias_in_bios")["train"]
    ds_test = load_dataset("LabHC/bias_in_bios")["test"]
    ds = concatenate_datasets([ds_train, ds_test])
    # 2. Construire un DataFrame pandas à partir des colonnes “profession” et “gender”
    df = pd.DataFrame({
        "profession": ds["profession"],
        "gender":     ds["gender"]
    })

    # 3. Table de contingence : nombre de codes 0/1 par profession
    ct = pd.crosstab(df["profession"], df["gender"], dropna=False)

    # 4. Renommer explicitement : 0→male, 1→female
    ct = ct.rename(columns={0: "male", 1: "female"})

    # 5. S’assurer que les deux colonnes existent
    for col in ("male", "female"):
        if col not in ct.columns:
            ct[col] = 0

    # 6. Totaux et pourcentages
    ct["total"]       = ct["male"] + ct["female"]
    ct["pct_male"]    = 100 * ct["male"]    / ct["total"]
    ct["pct_female"]  = 100 * ct["female"]  / ct["total"]

    # 7. Réinitialiser l’index et trier par taille de la profession (optionnel)
    result = (
        ct
        .reset_index()
        .sort_values("total", ascending=False)
        .loc[:, ["profession", "male", "pct_male", "female", "pct_female", "total"]]
    )

    # 8. Affichage
    pd.set_option("display.max_rows", None)
    print(result)

    # 9. Export CSV
    result.to_csv("stats_genre_par_profession.csv", index=False)
    print("\n–> Fichier 'stats_genre_par_profession.csv' créé.")

if __name__ == "__main__":
    main()
