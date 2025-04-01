import random
import os
from datasets import load_dataset

# Load the DBpedia 14 dataset
train_dataset = load_dataset("fancyzhx/dbpedia_14", split="train")
test_dataset = load_dataset("fancyzhx/dbpedia_14", split="test")

# Check the column names
print("Train dataset columns:", train_dataset.column_names)
print("Test dataset columns:", test_dataset.column_names)

# If the dataset does not have a "label_text" field, create one from the numerical "label"
if "label_text" not in train_dataset.column_names:
    def add_label_text(example):
        example["label_text"] = str(example["label"])
        return example
    train_dataset = train_dataset.map(add_label_text)
    test_dataset = test_dataset.map(add_label_text)

# Now, obtain the list of unique categories from the train dataset.
categories = list(set(train_dataset["label_text"]))
print("Unique categories:", categories)

# Shuffle and split categories into InD and OoD (60% for training InD)
random.shuffle(categories)
num_train_cats = int(0.6 * len(categories))
train_categories = categories[:num_train_cats]
ood_categories = categories[num_train_cats:]
print("InD categories:", train_categories)
print("OoD categories:", ood_categories)

# Create an output folder for the training files
train_folder = "dbpedia_env"
os.makedirs(train_folder, exist_ok=True)

def write_dataset_to_file(dataset, filename):
    # For DBpedia, assume the text field is "content".
    # If it is not, you can fall back to "text" or adjust as needed.
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example.get("content", example.get("text", ""))
            f.write(text.replace("\n", " ") + "\n")

# For each InD category, create a training file (one per environment)
for cat in train_categories:
    subset = train_dataset.filter(lambda x: x["label_text"] == cat)
    output_file = os.path.join(train_folder, f"{cat}.txt")
    write_dataset_to_file(subset, output_file)
    print(f"File for {cat} created with {len(subset)} examples.")

# Take 20% of the InD training data for validation
ind_val = train_dataset.filter(lambda x: x["label_text"] in train_categories).train_test_split(test_size=0.2, seed=42)["test"]
write_dataset_to_file(ind_val, os.path.join(train_folder, "val_ind.txt"))
print("Validation InD file created.")

# For OoD, filter the test set (or use a portion of the train set) for categories that are OoD
ood_val = test_dataset.filter(lambda x: x["label_text"] in ood_categories)
write_dataset_to_file(ood_val, os.path.join(train_folder, "val_ood.txt"))
print("Validation OoD file created.")
