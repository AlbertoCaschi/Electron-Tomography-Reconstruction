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
def save_reconstruction_progress(unet, diffusion, fixed_x_0, fixed_x_fbp, epoch, log_dir, device):
    """
    Runs the reverse diffusion process on a fixed validation sample and saves the plot.
    """
    unet.eval()

    x_0 = fixed_x_0.to(device, dtype=torch.float32)
    x_fbp = fixed_x_fbp.to(device, dtype=torch.float32)
    
    # Generation
    x_recon = diffusion.p_sample_loop(unet, x_fbp)
    
    x_0_vis = unnormalize_from_ddpm_range(x_0).squeeze().cpu().numpy()
    x_fbp_vis = unnormalize_from_ddpm_range(x_fbp).squeeze().cpu().numpy()
    x_recon_vis = unnormalize_from_ddpm_range(x_recon).squeeze().cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(x_0_vis, cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    axes[1].imshow(x_fbp_vis, cmap='gray')
    axes[1].set_title("Conditioning FBP (Artifacts)")
    axes[1].axis('off')
    
    axes[2].imshow(x_recon_vis, cmap='gray')
    axes[2].set_title(f"cDDPM Reconstruction (Epoch {epoch})")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    vis_dir = os.path.join(log_dir, "progress_images")
    os.makedirs(vis_dir, exist_ok=True)
    
    save_path = os.path.join(vis_dir, f"recon_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    default_log_path = "./cDDPM/logs/training_log.csv"
    plot_training_curves(default_log_path)