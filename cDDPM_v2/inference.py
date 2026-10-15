import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from PIL import Image

from cDDPM_v2.config import CONFIG
from cDDPM_v2.physics.operators import PyTorchTomographyOperator
from cDDPM_v2.models.unet import ConditionalUNet
from cDDPM_v2.models.diffusion import GaussianDiffusion


def normalize_to_ddpm_range(image):
    """Min-max scales an image to [-1, 1]."""
    img_min, img_max = image.min(), image.max()
    if img_max - img_min < 1e-6:
        return np.full_like(image, -1.0)
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

def center_crop(image, target_h, target_w):
    """Crops back-projected images from their padded diagonal size back to target dims."""
    H, W = image.shape
    if H > target_h or W > target_w:
        start_y = (H - target_h) // 2
        start_x = (W - target_w) // 2
        return image[start_y:start_y+target_h, start_x:start_x+target_w]
    return image

def process_and_reconstruct(unet, test_file, acquisition_config, device):
    """Handles the core data pipeline, physics operations, and model sampling."""
    # Initialize the GPU-native physics operator
    physics_operator = PyTorchTomographyOperator(CONFIG["physics"]["detector_pixels"], device)
    diffusion = GaussianDiffusion(CONFIG).to(device)

    # full angles for ground truth reconstruction
    raw_start, raw_end, raw_step = CONFIG["physics"]["raw_angles"]
    full_angles = np.arange(raw_start, raw_end + raw_step, raw_step)
    
    # load raw sinogram
    with mrcfile.open(test_file, permissive=True) as mrc:
        raw_sinogram = np.squeeze(mrc.data).astype(np.float32).copy()

    if raw_sinogram.shape[0] == len(full_angles):
        raw_sinogram = raw_sinogram.T

    # Convert to Tensor (B, C, W, num_angles) for PyTorch FBP
    raw_sino_tensor = torch.from_numpy(raw_sinogram).unsqueeze(0).unsqueeze(0).to(device)
    
    # Ground Truth FBP on GPU
    x_0_tensor = physics_operator.filtered_back_project(raw_sino_tensor, full_angles)
    x_0_np = x_0_tensor.squeeze().cpu().numpy()
    x_0_np = np.clip(x_0_np, a_min=0.0, a_max=None)
    
    target_h, target_w = CONFIG["data"]["image_dims"]
    
    pad_h, pad_w = max(0, target_h - x_0_np.shape[0]), max(0, target_w - x_0_np.shape[1])
    pad_top, pad_left = pad_h // 2, pad_w // 2

    x_0_padded = np.pad(
        x_0_np, 
        ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left)), 
        mode='constant',
        constant_values=0.0
    )

    # normalization
    g_min, g_max = x_0_padded.min(), x_0_padded.max()
    if g_max - g_min > 1e-6:
        x_0_padded = (x_0_padded - g_min) / (g_max - g_min)
    else:
        x_0_padded = np.zeros_like(x_0_padded)

    # noise thresholding
    x_0_padded = np.where(x_0_padded < CONFIG["data"]["noise_threshold"], 0.0, x_0_padded)
    
    # Convert padded GT back to GPU Tensor for forward projection
    x_0_padded_tensor = torch.from_numpy(x_0_padded).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    
    # Generate simulated sinogram natively on GPU of shape: (1, 1, W, num_angles)
    sinogram_compute_tensor = physics_operator.forward_project(x_0_padded_tensor, acquisition_config)
    
    # Keep sinogram strictly on GPU for returning later in the plot function
    sinogram_compute_np = sinogram_compute_tensor.squeeze().cpu().numpy()

    iterations = CONFIG["physics"]["SIRT_iterations"]
    
    # Generate SIRT Conditional Input natively on GPU
    print(f"Generating SIRT condition {iterations} on GPU...")
    x_sirt_raw_tensor = physics_operator.iterative_reconstruction(
        target_sinogram=sinogram_compute_tensor,
        angles_deg=acquisition_config,
        num_iterations=iterations,
        lr=1e-2  # Adjust if the SIRT conditioning looks too faint or too noisy
    )
    
    # Drop back to numpy for center cropping and DDPM normalization
    x_sirt_np = x_sirt_raw_tensor.squeeze().cpu().numpy()
    x_sirt_np = center_crop(x_sirt_np, target_h, target_w)
    x_sirt_np = np.ascontiguousarray(x_sirt_np, dtype=np.float32)

    # Convert back to tensor and scale to [-1, 1] for the diffusion model
    x_sirt_tensor = torch.from_numpy(normalize_to_ddpm_range(x_sirt_np)).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    
    # create geometry configuration tensor for inference
    current_max_tilt = np.abs(acquisition_config).max()
    num_projections = len(acquisition_config)
    acq_config_tensor = torch.tensor([current_max_tilt, num_projections], dtype=torch.float32).unsqueeze(0).to(device)

    # ==========================================
    # GPU-Native Uncertainty Map Computation
    # ==========================================
    print("Computing Uncertainty Map on GPU...")
    b_maps = []
    for i in range(num_projections):
        angle = torch.tensor([acquisition_config[i]], device=device, dtype=torch.float32)
        sino_slice = sinogram_compute_tensor[:, :, :, i:i+1] # Slice a single angle
        
        # Unfiltered back-projection
        b_i = physics_operator.back_project(sino_slice, angle)
        
        # Center crop the tensor
        H, W = b_i.shape[2], b_i.shape[3]
        if H > target_h or W > target_w:
            start_y, start_x = (H - target_h) // 2, (W - target_w) // 2
            b_i = b_i[:, :, start_y:start_y+target_h, start_x:start_x+target_w]
        b_maps.append(b_i)
        
    # Stack across a new dimension and calculate variance across all projections
    b_maps_tensor = torch.cat(b_maps, dim=0) # Shape: (num_projections, 1, H, W)
    variance_map = torch.var(b_maps_tensor, dim=0, keepdim=True) # Shape: (1, 1, H, W)
    
    # Min-Max normalize variance to [0, 1]
    var_min, var_max = variance_map.min(), variance_map.max()
    if var_max - var_min > 1e-8:
        u_map = (variance_map - var_min) / (var_max - var_min)
    else:
        u_map = torch.zeros_like(variance_map)
        
    # Shift to DDPM scale [-1, 1]
    uncertainty_tensor = (u_map * 2.0) - 1.0 
    # ==========================================

    use_projector = CONFIG["inference"]["use_projector_guidance"]
    
    print("Starting diffusion generation (may take a minute)...")
    with torch.no_grad():
        x_reconstructed_tensor = diffusion.p_sample_loop(
            unet, 
            x_sirt_tensor,
            acq_config_tensor,
            true_sinogram=sinogram_compute_tensor if use_projector else None,
            physics_op=physics_operator if use_projector else None,
            angles=acquisition_config if use_projector else None,
            uncertainty_map=uncertainty_tensor
        )
        
    # normalize back to visualize
    x_sirt_vis = unnormalize_from_ddpm_range(x_sirt_tensor).squeeze().cpu().numpy()
    x_recon_vis = unnormalize_from_ddpm_range(x_reconstructed_tensor).squeeze().cpu().numpy()

    return x_0_padded, sinogram_compute_np, x_sirt_vis, x_recon_vis

def plot_results(x_0_padded, sinogram, x_sirt_vis, x_recon_vis, max_angle, save_path=None):
    """Plots and optionally saves the 4-panel comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(x_0_padded, cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    axes[1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1].set_title(f"Masked Sinogram\nWedge: {max_angle}°")
    axes[1].axis('off')
    
    axes[2].imshow(x_sirt_vis, cmap='gray')
    axes[2].set_title("Conditioning SIRT\n")
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

def run_inference(checkpoint_path, test_file, acquisition_config):
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
    
    x_0_padded, sinogram, x_sirt_vis, x_recon_vis = process_and_reconstruct(
        unet, test_file, acquisition_config, device
    )

    plot_results(x_0_padded, sinogram, x_sirt_vis, x_recon_vis, max(acquisition_config))

def run_streamlit_inference(model, test_file, output_image_path, output_sirt_path, acquisition_config_dict):
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
    
    x_0_padded, sinogram, x_sirt_vis, x_recon_vis = process_and_reconstruct(
        unet, test_file, acquisition_config, device
    )

    # Save SIRT file
    os.makedirs(os.path.dirname(output_sirt_path), exist_ok=True)
    recon_8bit = x_recon_vis - np.min(x_recon_vis)
    if np.max(recon_8bit) > 0:
        recon_8bit = (recon_8bit / np.max(recon_8bit) * 255).astype(np.uint8)
    else:
        recon_8bit = recon_8bit.astype(np.uint8)

    # We are saving it to output_sirt_path
    Image.fromarray(recon_8bit).save(output_sirt_path)

    # Save plot
    plot_results(
        x_0_padded, sinogram, x_sirt_vis, x_recon_vis, 
        max(acquisition_config), save_path=output_image_path
    )


if __name__ == "__main__":
    CHECKPOINT = os.path.join(CONFIG["training"]["output_dir"], "unet_checkpoint_best.pt")
    TEST_FILE = r"./assets/rect_oval.mrc" #r".\cDDPM_v2\dataset\synthetic_raw\synthetic_sino_0000.mrc"
    ACQUISITION_CONFIG = np.arange(-40, 41, 5) # specific missing wedge and projection setup
    
    run_inference(CHECKPOINT, TEST_FILE, ACQUISITION_CONFIG)