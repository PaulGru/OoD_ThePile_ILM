import re
import wandb
import pandas as pd
import matplotlib.pyplot as plt

# 1. Connexion
api = wandb.Api()
project_path = "paul-grunenwaldplecy-institut-polytechnique-de-paris/invariant-language-modeling"

# 2. Récupère tous les runs
runs = api.runs(project_path)

# 3. Filtre par nom avec regex
pattern = re.compile(r"^model_runs_(elm|ilm)_lr5e-05_seed([0-3])_eval$")
filtered = []
for run in runs:
    m = pattern.match(run.name)
    if not m:
        continue
    model = m.group(1)         # 'elm' ou 'ilm'
    seed  = int(m.group(2))    # 0,1,2 ou 3
    filtered.append((run, model, seed))

if not filtered:
    raise RuntimeError("Aucun run ne correspond à la regex — vérifie les noms de tes runs.")

# 4. Charge l’historique
records = []
for run, model, seed in filtered:
    hist = run.history(
        keys=["_step", "eval_ood/perplexity", "eval_in/perplexity"],
        pandas=True
    )
    if hist.empty:
        continue
    hist = hist.rename(columns={"_step": "step"})
    hist["model"] = model
    hist["seed"]  = seed
    records.append(hist[["model","seed","step","eval_ood/perplexity","eval_in/perplexity"]])

if not records:
    raise RuntimeError("Les historiques sont vides ! Vérifie que tu logs bien ces clés en mode `history` et non `summary`.")

# 5. Concatène et moyenne
df       = pd.concat(records, ignore_index=True)
mean_df  = (
    df
    .groupby(["model","step"], as_index=False)
    [["eval_ood/perplexity","eval_in/perplexity"]]
    .mean()
)

# 6. Tracé et enregistrement
plt.figure(figsize=(8,5))
for model, grp in mean_df.groupby("model"):
    plt.plot(grp["step"], grp["eval_ood/perplexity"], label=f"{model} – OOD")
    plt.plot(grp["step"], grp["eval_in/perplexity"], label=f"{model} – IN")

plt.xlabel("Step")
plt.ylabel("Perplexity")
plt.title("Moyenne des perplexités IN vs OOD sur seeds 0–3 pour lr=5e-05")
plt.legend()
plt.tight_layout()

# Sauvegarde PNG
output_path = "perplexity_mean_elm_ilm_lr5e-05.png"
plt.savefig(output_path, dpi=300)
print(f"✅ Graphique enregistré sous : {output_path}")

# (Optionnel) affiche à l’écran
plt.show()
