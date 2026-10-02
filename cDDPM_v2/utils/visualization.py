import os
import csv
import torch
import matplotlib.pyplot as plt

def plot_training_curves(csv_path):
    """
    Reads the training log CSV and plots the Train and Validation losses.
    
    Args:
        csv_path (str): The path to the training_log.csv file.
    """
    if not os.path.exists(csv_path):
        print(f"\n[Error] Log file not found at {csv_path}. Cannot generate plot.")
        return

    epochs = []
    train_losses = []
    val_losses = []

    # Read the CSV data
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row["Epoch"]))
                train_losses.append(float(row["Train Loss"]))
                val_losses.append(float(row["Val Loss"]))
            except ValueError:
                # Skip rows with incomplete or malformed data
                continue

    if not epochs:
        print("\n[Warning] No valid data found in the CSV to plot.")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', linewidth=2, marker='o', markersize=4)
    
    plt.title('Diffusion Model: Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # save
    plot_path = csv_path.replace('.csv', '.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\n--> Loss plot saved to: {plot_path}")
    
    # display
    # plt.show()


def unnormalize_from_ddpm_range(tensor):
    """Converts a [-1, 1] tensor back to [0, 1] for visualization."""
    tensor = torch.clamp(tensor, -1.0, 1.0)
    return (tensor + 1.0) / 2.0

@torch.no_grad()
def save_loss_plot(train_losses, log_dir, val_losses=None):
    """
    Save training/validation losses using:
    1. Linear scale
    2. Logarithmic scale
    """
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # -------------------------
    # Linear scale plot
    # -------------------------
    axes[0].plot(epochs, train_losses,
    label="Training Loss",
    color="tab:blue",
    linewidth=2)
    
    if val_losses is not None:
        axes[0].plot(epochs, val_losses,
        label="Validation Loss",
        color="tab:orange",
        linewidth=2)
    
    axes[0].set_title("Loss (Linear Scale)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # -------------------------
    # Log scale plot
    # -------------------------
    axes[1].plot(epochs, train_losses,
    label="Training Loss",
    color="tab:blue",
    linewidth=2)
    
    if val_losses is not None:
        axes[1].plot(epochs, val_losses,
        label="Validation Loss",
        color="tab:orange",
        linewidth=2)
    
    axes[1].set_title("Loss (Log Scale)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale('log')
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    save_path = os.path.join(log_dir, "loss_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    default_log_path = "./cDDPM_v2/logs/training_log.csv"
    plot_training_curves(default_log_path)