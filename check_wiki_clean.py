from datasets import load_dataset

dataset = load_dataset(
    "text",
    data_files={
        "train": "Wiki/train/all_train.txt",
        "validation": "Wiki/val_env/val_ind.txt",
    }
)

print(dataset)
