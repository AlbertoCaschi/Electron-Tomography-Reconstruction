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

def run_inference(checkpoint_path, test_file, acquisition_config):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # model
    print("Loading models and physics operators...")
    physics_operator = TomographyOperator(CONFIG["physics"])
    unet = ConditionalUNet(CONFIG).to(device)
    diffusion = GaussianDiffusion(CONFIG).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    print(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}.")
    

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
    
    pad_h = max(0, target_h - x_0_np.shape[0])
    pad_w = max(0, target_w - x_0_np.shape[1])
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # padding: 362x362 -> 368x368
    x_0_padded = np.pad(
        x_0_np, 
        ((pad_top, pad_bottom), (pad_left, pad_right)), 
        mode='constant',
        constant_values=0
    )
    
    print("Simulating limited-angle measurement...")
    sinogram = physics_operator.forward_project(x_0_padded, acquisition_config) # full FBP -> full sinogram
    x_fbp_np = physics_operator.filtered_back_project(sinogram, acquisition_config) # incomplete sinogram -> FBP with artifacts
    
    # normalize and convert to torch tensors
    x_fbp_tensor = torch.from_numpy(normalize_to_ddpm_range(x_fbp_np)).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    

    ## Reconstruction
    print("Starting diffusion generation (may take a minute)...")
    with torch.no_grad():
        x_reconstructed_tensor = diffusion.p_sample_loop(unet, x_fbp_tensor)
        
    # normalize back to visualize
    x_fbp_vis = unnormalize_from_ddpm_range(x_fbp_tensor).squeeze().cpu().numpy()
    x_recon_vis = unnormalize_from_ddpm_range(x_reconstructed_tensor).squeeze().cpu().numpy()


    ## Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(x_0_padded, cmap='gray')
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



def run_streamlit_inference(
        checkpoint_path,
        test_file,
        output_image_path,
        output_fbp_path,
        acquisition_config
        ):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # model
    print("Loading models and physics operators...")
    physics_operator = TomographyOperator(CONFIG["physics"])
    unet = ConditionalUNet(CONFIG).to(device)
    diffusion = GaussianDiffusion(CONFIG).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    print(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}.")
    

    # full angles for ground truth reconstruction
    raw_start, raw_end, raw_step = CONFIG["physics"]["raw_angles"]
    full_angles = np.arange(raw_start, raw_end + raw_step, raw_step)

    acquisition_config = np.arange(acquisition_config["range"][0], acquisition_config["range"][1]+1, acquisition_config["step"])
    
    # raw sinogram -> ground truth -> simulated artifacts
    with mrcfile.open(test_file, permissive=True) as mrc:
        raw_sinogram = np.squeeze(mrc.data).astype(np.float32).copy()

    if raw_sinogram.shape[0] == len(full_angles):
        raw_sinogram = raw_sinogram.T

    x_0_np = physics_operator.filtered_back_project(raw_sinogram, full_angles)
        
    target_h, target_w = CONFIG["data"]["image_dims"]
    
    pad_h = max(0, target_h - x_0_np.shape[0])
    pad_w = max(0, target_w - x_0_np.shape[1])
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # padding: 362x362 -> 368x368
    x_0_padded = np.pad(
        x_0_np, 
        ((pad_top, pad_bottom), (pad_left, pad_right)), 
        mode='constant',
        constant_values=0
    )
    
    print("Simulating limited-angle measurement...")
    sinogram = physics_operator.forward_project(x_0_padded, acquisition_config) # full FBP -> full sinogram
    x_fbp_np = physics_operator.filtered_back_project(sinogram, acquisition_config) # incomplete sinogram -> FBP with artifacts
    
    # normalize and convert to torch tensors
    x_fbp_tensor = torch.from_numpy(normalize_to_ddpm_range(x_fbp_np)).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    

    ## Reconstruction
    print("Starting diffusion generation (may take a minute)...")
    with torch.no_grad():
        x_reconstructed_tensor = diffusion.p_sample_loop(unet, x_fbp_tensor)
        
    # normalize back to visualize
    x_fbp_vis = unnormalize_from_ddpm_range(x_fbp_tensor).squeeze().cpu().numpy()
    x_recon_vis = unnormalize_from_ddpm_range(x_reconstructed_tensor).squeeze().cpu().numpy()
    

    # save FBP file
    os.makedirs(os.path.dirname(output_fbp_path), exist_ok=True)
    recon_8bit = x_recon_vis - np.min(x_recon_vis)
    if np.max(recon_8bit) > 0:
        recon_8bit = (recon_8bit / np.max(recon_8bit) * 255).astype(np.uint8)
    else:
        recon_8bit = recon_8bit.astype(np.uint8)
    im = Image.fromarray(recon_8bit)
    im.save(output_fbp_path)


    ## Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(x_0_padded, cmap='gray')
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
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=200, bbox_inches='tight')
    plt.close(fig)



if __name__ == "__main__":

    CHECKPOINT = os.path.join(CONFIG["training"]["output_dir"], "unet_checkpoint_best.pt")
    TEST_FILE = r"C:\Users\Alberto\Desktop\Electron-Tomography-Reconstruction\cDDPM\dataset\test_data\2_squares.mrc"
    ACQUISITION_CONFIG = np.arange(-50, 51, 5) # specific missing wedge and projection setup
    
    run_inference(CHECKPOINT, TEST_FILE, ACQUISITION_CONFIG)
