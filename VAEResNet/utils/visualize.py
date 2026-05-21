import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def to_numpy(tensor):
    """Safely converts a PyTorch tensor to a 2D NumPy array for plotting."""
    if isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        arr = tensor.detach().cpu().float().numpy()
        
    # Squeeze out batch and channel dimensions if present (e.g., from [1, 1, H, W] to [H, W])
    return np.squeeze(arr)

def plot_sinogram_comparison(input_sino, target_sino, pred_sino, epoch=None, save_path=None):
    """
    Plots the masked input, the ground truth target, and the VAE prediction side-by-side.
    """
    in_np = to_numpy(input_sino)
    target_np = to_numpy(target_sino)
    pred_np = to_numpy(pred_sino)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use aspect='auto' because angle steps and detector pixels often have very different scales
    im0 = axes[0].imshow(in_np, cmap='gray', aspect='auto')
    axes[0].set_title("Input (Masked/Noisy)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(target_np, cmap='gray', aspect='auto')
    axes[1].set_title("Target (Complete)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pred_np, cmap='gray', aspect='auto')
    axes[2].set_title("VAE Prediction")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Detector Pixels")
        ax.set_ylabel("Projection Angles")

    if epoch is not None:
        fig.suptitle(f"Sinogram Reconstruction - Epoch {epoch}", fontsize=16)

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_reconstruction_dashboard(input_sino, target_sino, pred_sino, 
                                  recon_input, recon_target, recon_pred, 
                                  epoch=None, save_path=None):
    """
    Creates a comprehensive 2x3 dashboard. 
    Top row: Sinograms.
    Bottom row: Corresponding 2D FBP/SART Reconstructions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # --- Top Row: Sinograms ---
    sinos = [to_numpy(input_sino), to_numpy(target_sino), to_numpy(pred_sino)]
    titles_top = ["Input Sinogram (Missing Wedge)", "Target Sinogram (Complete)", "Predicted Sinogram"]
    
    for i in range(3):
        im = axes[0, i].imshow(sinos[i], cmap='gray', aspect='auto')
        axes[0, i].set_title(titles_top[i])
        axes[0, i].set_ylabel("Angles")
        axes[0, i].set_xlabel("Detector")
        fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

    # --- Bottom Row: 2D Reconstructions ---
    recons = [to_numpy(recon_input), to_numpy(recon_target), to_numpy(recon_pred)]
    titles_bot = ["Recon from Input (Artifacts)", "Recon from Target (Ground Truth)", "Recon from Prediction"]
    
    for i in range(3):
        # aspect='equal' is crucial here to preserve the physical geometry of the 2D slice
        im = axes[1, i].imshow(recons[i], cmap='gray', aspect='equal')
        axes[1, i].set_title(titles_bot[i])
        axes[1, i].axis('off') # Hide axes for clean image viewing
        fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    if epoch is not None:
        fig.suptitle(f"Tomography VAE Dashboard - Epoch {epoch}", fontsize=18)

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_training_curves(train_losses, val_losses, beta_values=None, save_path=None):
    """
    Plots training and validation losses, and optionally the KL Beta schedule.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(train_losses, label='Train Total Loss', color='tab:blue', linestyle='-')
    ax1.plot(val_losses, label='Val Total Loss', color='tab:cyan', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    if beta_values is not None:
        ax2 = ax1.twinx()  
        ax2.set_ylabel('KL Beta', color='tab:red')  
        ax2.plot(beta_values, label='Beta Schedule', color='tab:red', linestyle=':')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title("Training Metrics")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()