import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from PIL import Image

from cDDPM.config import CONFIG
from cDDPM.physics.operators import TomographyOperator
from cDDPM.models.unet import ConditionalUNet
from cDDPM.models.diffusion import GaussianDiffusion


def normalize_to_ddpm_range(image):
    """Min-max scales an image to [-1, 1]."""
    img_min, img_max = image.min(), image.max()
    if img_max - img_min < 1e-6:
        return np.zeros_like(image)
    img_normalized = (image - img_min) / (img_max - img_min)
    return (img_normalized * 2.0) - 1.0

def unnormalize_from_ddpm_range(tensor):
    """Converts a [-1, 1] tensor back to [0, 1] for visualization."""
    tensor = torch.clamp(tensor, -1.0, 1.0)
    return (tensor + 1.0) / 2.0

def get_device():
    """Returns the optimal available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def process_and_reconstruct(unet, test_file, acquisition_config, device):
    """Handles the core data pipeline, physics operations, and model sampling."""
    physics_operator = TomographyOperator(CONFIG["physics"])
    diffusion = GaussianDiffusion(CONFIG).to(device)

    # full angles for ground truth reconstruction
    raw_start, raw_end, raw_step = CONFIG["physics"]["raw_angles"]
    full_angles = np.arange(raw_start, raw_end + raw_step, raw_step)
    
    # raw sinogram -> ground truth -> simulated artifacts
    with mrcfile.open(test_file, permissive=True) as mrc:
        raw_sinogram = np.squeeze(mrc.data).astype(np.float32).copy()

    if raw_sinogram.shape[0] == len(full_angles):
        raw_sinogram = raw_sinogram.T

    x_0_np = physics_operator.filtered_back_project(raw_sinogram, full_angles)
        
    target_h, target_w = CONFIG["data"]["image_dims"]
    
    pad_h, pad_w = max(0, target_h - x_0_np.shape[0]), max(0, target_w - x_0_np.shape[1])
    pad_top, pad_left = pad_h // 2, pad_w // 2
    
    x_0_padded = np.pad(
        x_0_np, 
        ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left)), 
        mode='constant',
        constant_values=0
    )

    # normalization
    g_min, g_max = x_0_padded.min(), x_0_padded.max()
    if g_max - g_min > 1e-6:
        x_0_padded = (x_0_padded - g_min) / (g_max - g_min)
    else:
        x_0_padded = np.zeros_like(x_0_padded)

    # noise thresholding
    x_0_padded = np.where(x_0_padded < CONFIG["data"]["noise_threshold"], 0.0, x_0_padded)
    
    # Generate sinogram
    sinogram_compute = physics_operator.forward_project(x_0_padded, acquisition_config)
    x_fbp_np = physics_operator.filtered_back_project(sinogram_compute, acquisition_config)
    
    # normalize and convert to torch tensors
    x_fbp_tensor = torch.from_numpy(normalize_to_ddpm_range(x_fbp_np)).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    
    use_projector = CONFIG["inference"]["use_projector_guidance"]

    print("Starting diffusion generation (may take a minute)...")
    with torch.no_grad():
        x_reconstructed_tensor = diffusion.p_sample_loop(
            unet, 
            x_fbp_tensor,
            true_sinogram=sinogram_compute if use_projector else None,
            physics_op=physics_operator if use_projector else None,
            angles=acquisition_config if use_projector else None
        )
        
    # normalize back to visualize
    x_fbp_vis = unnormalize_from_ddpm_range(x_fbp_tensor).squeeze().cpu().numpy()
    x_recon_vis = unnormalize_from_ddpm_range(x_reconstructed_tensor).squeeze().cpu().numpy()

    return x_0_padded, sinogram_compute, x_fbp_vis, x_recon_vis

def plot_results(x_0_padded, sinogram, x_fbp_vis, x_recon_vis, max_angle, save_path=None):
    """Plots and optionally saves the 4-panel comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(x_0_padded, cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    axes[1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1].set_title(f"Masked Sinogram\nWedge: {max_angle}°")
    axes[1].axis('off')
    
    axes[2].imshow(x_fbp_vis, cmap='gray')
    axes[2].set_title("Conditioning FBP\n")
    axes[2].axis('off')
    
    axes[3].imshow(x_recon_vis, cmap='gray')
    axes[3].set_title("cDDPM Reconstruction\n")
    axes[3].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def run_inference(checkpoint_path, test_file, acquisition_config, noise_threshold=0.3):
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading models and physics operators...")
    unet = ConditionalUNet(CONFIG).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    print(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}.")
    
    x_0_padded, sinogram, x_fbp_vis, x_recon_vis = process_and_reconstruct(
        unet, test_file, acquisition_config, device, noise_threshold
    )

    plot_results(x_0_padded, sinogram, x_fbp_vis, x_recon_vis, max(acquisition_config))

def run_streamlit_inference(model, test_file, output_image_path, output_fbp_path, acquisition_config_dict, noise_threshold=0.3):
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading models and physics operators...")
    unet = model.to(device)
    unet.eval()

    # Parse acquisition config dict to numpy array
    acquisition_config = np.arange(
        acquisition_config_dict["range"][0], 
        acquisition_config_dict["range"][1]+1, 
        acquisition_config_dict["step"]
    )
    
    x_0_padded, sinogram, x_fbp_vis, x_recon_vis = process_and_reconstruct(
        unet, test_file, acquisition_config, device, noise_threshold
    )

    # Save FBP file
    os.makedirs(os.path.dirname(output_fbp_path), exist_ok=True)
    recon_8bit = x_recon_vis - np.min(x_recon_vis)
    if np.max(recon_8bit) > 0:
        recon_8bit = (recon_8bit / np.max(recon_8bit) * 255).astype(np.uint8)
    else:
        recon_8bit = recon_8bit.astype(np.uint8)
    
    Image.fromarray(recon_8bit).save(output_fbp_path)

    # Save plot
    plot_results(
        x_0_padded, sinogram, x_fbp_vis, x_recon_vis, 
        max(acquisition_config), save_path=output_image_path
    )


if __name__ == "__main__":
    CHECKPOINT = os.path.join(CONFIG["training"]["output_dir"], "cDDPM.pt")
    TEST_FILE = r".\assets\2_squares.mrc"
    ACQUISITION_CONFIG = np.arange(-50, 51, 5) # specific missing wedge and projection setup
    
    run_inference(CHECKPOINT, TEST_FILE, ACQUISITION_CONFIG)