from transformers import DistilBertTokenizer, DistilBertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
import math
import pandas as pd
import os

# 1. Charger tokenizer et modèle
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

# 2. Charger datasets
def load_texts_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip() != ""]
    return lines

train_path = "Pile_envs/train_env/all_train.txt"
eval_path = "Pile_envs/val_env/val_ind.txt"
ood_path = "Pile_envs/val_env/val_ood.txt"

all_train = load_texts_from_file(train_path)
all_eval = load_texts_from_file(eval_path)
all_ood = load_texts_from_file(ood_path)

def clean_texts(texts):
    cleaned = []
    for text in texts:
        text = text.strip()
        if len(text) > 0 and len(text.split()) >= 3:  # minimum 3 mots
            cleaned.append(text)
    return cleaned

all_train = clean_texts(all_train)
all_eval = clean_texts(all_eval)
all_ood = clean_texts(all_ood)

train_dataset = Dataset.from_dict({"text": all_train})
eval_dataset = Dataset.from_dict({"text": all_eval})
ood_dataset = Dataset.from_dict({"text": all_ood})


# 3. Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=16)
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=16)
ood_dataset = ood_dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=16)

# 4. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# 5. Arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./distilbert-mlm-finetuned",
    overwrite_output_dir=True,
    evaluation_strategy="steps",        
    save_strategy="steps",            
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_steps=1000,
    per_device_train_batch_size=64,
    logging_steps=100,
    prediction_loss_only=False,
    fp16=False,
    learning_rate=1e-4,
    max_grad_norm=1.0,
)

# 7. Callback pour logger toutes les steps dans un CSV
class LogToCSVCallback(TrainerCallback):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step_data = {"step": state.global_step}

            # Enregistrer la perte d'entraînement si disponible
            if "loss" in logs:
                step_data["train_loss"] = logs["loss"]

            # Enregistrer la perte de validation si disponible
            if "eval_loss" in logs:
                step_data["eval_loss"] = logs["eval_loss"]
                step_data["perplexity"] = math.exp(logs["eval_loss"])  # calcul de perplexité à la volée

            self.logs.append(step_data)

            # Sauvegarder immédiatement pour ne rien perdre en cas d'interruption
            df = pd.DataFrame(self.logs)
            df.to_csv(self.csv_path, index=False)

# 8. Instancier Trainer
csv_logger = LogToCSVCallback(csv_path="./training_logs.csv")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[csv_logger],
)

# 9. Lancer l'entraînement
trainer.train()

import math
import pandas as pd

# Évaluer sur InD (jeu de validation)
ind_results = trainer.evaluate(eval_dataset=eval_dataset)
ind_loss = ind_results["eval_loss"]
ind_perplexity = math.exp(ind_loss)

print(f"Résultats sur InD :")
print(f"  - Loss : {ind_loss:.4f}")
print(f"  - Perplexity : {ind_perplexity:.2f}")

# Évaluer sur OoD (autre dataset)
ood_results = trainer.evaluate(eval_dataset=ood_dataset)
ood_loss = ood_results["eval_loss"]
ood_perplexity = math.exp(ood_loss)

print(f"Résultats sur OoD :")
print(f"  - Loss : {ood_loss:.4f}")
print(f"  - Perplexity : {ood_perplexity:.2f}")

# Sauvegarder dans un CSV final
final_metrics = {
    "dataset": ["InD", "OoD"],
    "loss": [ind_loss, ood_loss],
    "perplexity": [ind_perplexity, ood_perplexity],
}
df_final = pd.DataFrame(final_metrics)
df_final.to_csv("./final_results.csv", index=False)
