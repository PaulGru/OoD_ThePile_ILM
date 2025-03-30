import csv
import matplotlib.pyplot as plt

def plot_learning_curve(log_path="training_log.csv", output_path=None):
    epochs = []
    train_losses = []
    # Read the CSV log file
    with open(log_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Parse epoch and training loss
            if row.get("train_loss"):
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
    # Plot the training loss curve
    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label="Training Loss")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True)
    plt.legend()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_validation_loss(log_path="training_log.csv", output_path=None):
    # Similar implementation: read epochs and val_loss, then plot
    epochs = []
    val_losses = []
    with open(log_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Only include epochs where validation loss is available (non-empty)
            if row.get("val_loss") not in (None, "", "NA"):
                epochs.append(int(row["epoch"]))
                val_losses.append(float(row["val_loss"]))
    plt.figure()
    plt.plot(epochs, val_losses, marker='o', color='orange', label="Validation Loss")
    plt.title("Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.legend()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_perplexity(log_path="training_log.csv", output_path=None):
    epochs = []
    perplexities = []
    with open(log_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("perplexity") not in (None, "", "NA"):
                epochs.append(int(row["epoch"]))
                perplexities.append(float(row["perplexity"]))
    plt.figure()
    plt.plot(epochs, perplexities, marker='o', color='green', label="Perplexity")
    plt.title("Perplexity over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.grid(True)
    plt.legend()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Quick test: if run directly, display all three plots (ensure training_log.csv exists in the same directory)
    plot_learning_curve()
    plot_validation_loss()
    plot_perplexity()