import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import mrcfile

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

def run_inference(checkpoint_path, test_file, acquisition_config):
    """
    Runs the inference pipeline to reconstruct a 2D slice from a simulated missing wedge.
    """
    # 1. Device Configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # 2. Model & Physics Initialization
    print("Loading models and physics operators...")
    physics_operator = TomographyOperator(CONFIG["physics"])
    unet = ConditionalUNet(CONFIG).to(device)
    diffusion = GaussianDiffusion(CONFIG).to(device)
    
    # Load trained weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    print(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}.")
    
    
    with mrcfile.open(test_file, permissive=True) as mrc:
        x_0_np = np.squeeze(mrc.data).astype(np.float32).copy()
        
    target_h, target_w = CONFIG["data"]["image_dims"]
    
    pad_h = max(0, target_h - x_0_np.shape[0])
    pad_w = max(0, target_w - x_0_np.shape[1])
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Pad the ground truth to 368x368
    x_0_padded = np.pad(
        x_0_np, 
        ((pad_top, pad_bottom), (pad_left, pad_right)), 
        mode='constant', 
        constant_values=0
    )
    # ------------------------------
        
    # Pick a specific acquisition geometry for testing (e.g., -40 to 40 degrees, step 5)
    
    
    # Run the forward physics to get our conditioning FBP image using the PADDED image
    print("Simulating limited-angle measurement...")
    sinogram = physics_operator.forward_project(x_0_padded, acquisition_config)
    
    # Note: Masking step removed because the limited angles above natively create the missing wedge
    x_fbp_np = physics_operator.filtered_back_project(sinogram, acquisition_config)
    
    # Normalize to [-1, 1] and convert to PyTorch tensors of shape [1, 1, H, W]
    x_fbp_tensor = torch.from_numpy(normalize_to_ddpm_range(x_fbp_np)).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    
    # 4. Generative Reconstruction (The Reverse Process)
    print("Starting diffusion generation. This may take a minute...")
    with torch.no_grad():
        # Call the p_sample_loop from diffusion.py
        x_reconstructed_tensor = diffusion.p_sample_loop(unet, x_fbp_tensor)
        
    # 5. Post-processing and Visualization
    # Move tensors back to CPU and un-normalize for matplotlib
    x_fbp_vis = unnormalize_from_ddpm_range(x_fbp_tensor).squeeze().cpu().numpy()
    x_recon_vis = unnormalize_from_ddpm_range(x_reconstructed_tensor).squeeze().cpu().numpy()
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(x_0_np, cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    axes[1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1].set_title(f"Masked Sinogram\nWedge: {max(acquisition_config)}°")
    axes[1].axis('off')
    
    axes[2].imshow(x_fbp_vis, cmap='gray')
    axes[2].set_title("Conditioning FBP\n")
    axes[2].axis('off')
    
    axes[3].imshow(x_recon_vis, cmap='gray')
    axes[3].set_title("cDDPM Reconstruction\n")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":

    LATEST_CHECKPOINT = os.path.join(CONFIG["training"]["output_dir"], "unet_checkpoint_best.pt")
    TEST_FILE = r"C:\Users\Alberto\Desktop\Electron-Tomography-Reconstruction\cDDPM\dataset\test_data\2_squares.mrc"
    ACQUISITION_CONFIG = np.arange(-50, 51, 5)
    
    try:
        run_inference(LATEST_CHECKPOINT, TEST_FILE, ACQUISITION_CONFIG)
    except FileNotFoundError as e:
        print(e)
        print("Train the model first using 'python train.py' to generate a checkpoint.")