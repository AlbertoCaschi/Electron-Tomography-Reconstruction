import os
import csv
import torch
import matplotlib.pyplot as plt

def plot_training_curves(csv_path):
    """
    Reads the training log CSV and plots the Train and Validation losses.
    Generates two side-by-side plots: linear scale and logarithmic scale.
    
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

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Linear scale plot
    axes[0].plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2, marker='o', markersize=4)
    axes[0].plot(epochs, val_losses, label='Validation Loss', color='orange', linewidth=2, marker='o', markersize=4)
    
    axes[0].set_title('Diffusion Model: Training and Validation Loss (Linear)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('L1 Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Log scale plot
    axes[1].plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2, marker='o', markersize=4)
    axes[1].plot(epochs, val_losses, label='Validation Loss', color='orange', linewidth=2, marker='o', markersize=4)
    
    axes[1].set_title('Diffusion Model: Training and Validation Loss (Log Scale)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('L1 Loss')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout()

    plot_path = csv_path.replace('.csv', '.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\n--> Loss plot saved to: {plot_path}")

    plt.close(fig)


def unnormalize_from_ddpm_range(tensor):
    """Converts a [-1, 1] tensor back to [0, 1] for visualization."""
    tensor = torch.clamp(tensor, -1.0, 1.0)
    return (tensor + 1.0) / 2.0


@torch.no_grad()
def save_reconstruction_progress(unet, diffusion, x_0, x_sirt, x_unc, acq_config, epoch, log_dir, device):
    """Generates and saves a spatial image comparison during training."""
    unet.eval()
    
    # Send tensors to device
    x_sirt_dev = x_sirt.to(device, dtype=torch.float32)
    x_unc_dev = x_unc.to(device, dtype=torch.float32)
    acq_config_dev = acq_config.to(device, dtype=torch.float32)

    # Generate a single sample (bypassing projector guidance for quick visualization)
    x_recon = diffusion.p_sample_loop(
        unet, 
        x_sirt_dev, 
        acq_config_dev, 
        uncertainty_map=x_unc_dev,
        guidance_scale=1.0
    )
    
    # Unnormalize back to [0, 1] and convert to numpy
    x_0_np = unnormalize_from_ddpm_range(x_0).squeeze().cpu().numpy()
    x_sirt_np = unnormalize_from_ddpm_range(x_sirt).squeeze().cpu().numpy()
    x_recon_np = unnormalize_from_ddpm_range(x_recon).squeeze().cpu().numpy()
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(x_0_np, cmap='gray')
    axes[0].set_title("Ground Truth")
    
    axes[1].imshow(x_sirt_np, cmap='gray')
    axes[1].set_title("SIRT Conditioning")
    
    axes[2].imshow(x_recon_np, cmap='gray')
    axes[2].set_title(f"Reconstruction (Epoch {epoch})")
    
    for ax in axes:
        ax.axis('off')
        
    save_path = os.path.join(log_dir, f"recon_epoch_{epoch}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    unet.train()


if __name__ == "__main__":
    default_log_path = "./cDDPM_v2/logs/training_log.csv"
    plot_training_curves(default_log_path)