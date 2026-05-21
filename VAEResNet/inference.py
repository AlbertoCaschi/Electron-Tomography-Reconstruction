import os
import torch
import numpy as np
import mrcfile
import matplotlib.pyplot as plt

# Import custom modules
from models.vae import TomographyVAE
from utils.reconstruction import reconstruct_fbp_single
from utils.visualize import to_numpy

def load_model(checkpoint_path, config, device):
    """Loads the VAE model from a saved checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    model = TomographyVAE(
        latent_dim=config['latent_dim'], 
        target_size=config['target_size'], 
        resnet_type=config['resnet_type'],
        freeze_early_layers=False 
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() 
    return model

def load_and_preprocess_mrc(file_path, target_size=(181, 512)):
    """Loads a single .mrc sinogram and normalizes it."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
        
    with mrcfile.open(file_path, permissive=True) as mrc:
        data = mrc.data.copy().astype(np.float32)
        
    if data.ndim == 3 and data.shape[0] == 1:
        data = np.squeeze(data, axis=0)
        
    if data.shape[0] == target_size[1]:
        data = data.T
        
    assert data.shape == target_size, f"Expected {target_size}, got {data.shape}"
    
    # Normalize to [0,1]
    d_min, d_max = data.min(), data.max()
    if d_max - d_min > 1e-6:
        data = (data - d_min) / (d_max - d_min)
        
    return data

def save_mrc(data, file_path):
    """Saves a numpy array back to an .mrc file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with mrcfile.new(file_path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))

def apply_missing_wedge_mask(sino, tilt_range, step, base_angles_deg):
    """Zeros out all rows that do not fall on the acquired angles."""
    masked_sino = np.zeros_like(sino)
    min_angle, max_angle = tilt_range
    
    acquired_angles = np.arange(min_angle, max_angle + 1e-5, step)
    
    for angle in acquired_angles:
        idx = np.argmin(np.abs(base_angles_deg - angle))
        masked_sino[idx, :] = sino[idx, :]
        
    return masked_sino

def run_inference(input_mrc_path, checkpoint_path, output_dir, mask_config=None, full_angle_range=(-90, 90)):
    """
    Main inference pipeline.
    If mask_config is provided, it simulates an incomplete acquisition from a complete sinogram.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running inference on {device}...")
    os.makedirs(output_dir, exist_ok=True)
    
    inference_config = {
        'latent_dim': 1024,
        'target_size': (181, 362), 
        'resnet_type': 'resnet18'
    }
    
    base_angles_deg = np.linspace(full_angle_range[0], full_angle_range[1], inference_config['target_size'][0])
    
    model = load_model(checkpoint_path, inference_config, device)
    
    print(f"Processing input file: {input_mrc_path}")
    target_sino_np = load_and_preprocess_mrc(input_mrc_path, target_size=inference_config['target_size'])
    
    if mask_config:
        print(f"Applying mask: Range {mask_config['range']}, Step {mask_config['step']}°")
        network_input_np = apply_missing_wedge_mask(target_sino_np, mask_config['range'], mask_config['step'], base_angles_deg)
    else:
        print("No mask config provided. Using input as-is.")
        network_input_np = target_sino_np.copy()
    
    input_tensor = torch.from_numpy(network_input_np).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon_x, _, _ = model(input_tensor) 
    pred_sino_np = to_numpy(recon_x) 
    
    base_name = os.path.basename(input_mrc_path).replace('.mrc', '')
    output_sino_path = os.path.join(output_dir, f"{base_name}_inpainted.mrc")
    save_mrc(pred_sino_np, output_sino_path)
    
    print("Reconstructing 2D images via Filtered Back Projection...")
    recon_target = reconstruct_fbp_single(target_sino_np, base_angles_deg, filter_name='ramp')
    recon_masked = reconstruct_fbp_single(network_input_np, base_angles_deg, filter_name='ramp')
    recon_pred = reconstruct_fbp_single(pred_sino_np, base_angles_deg, filter_name='ramp')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    sinos = [target_sino_np, network_input_np, pred_sino_np]
    titles_top = ["Ground Truth (Complete)", "Masked Input to VAE", "VAE Prediction"]
    for i in range(3):
        axes[0, i].imshow(sinos[i], cmap='gray', aspect='auto')
        axes[0, i].set_title(titles_top[i])
        axes[0, i].set_ylabel("Angles")
        axes[0, i].set_xlabel("Detector Pixels")

    recons = [recon_target, recon_masked, recon_pred]
    titles_bot = ["Perfect 2D Reconstruction", "Degraded 2D (Artifacts)", "Restored 2D (Inpainted)"]
    for i in range(3):
        axes[1, i].imshow(recons[i], cmap='gray', aspect='equal')
        axes[1, i].set_title(titles_bot[i])
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{base_name}_inference_dashboard.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Saved inference dashboard to: {plot_path}")
    print("Inference Complete!")

if __name__ == "__main__":
    CHECKPOINT_FILE = "./checkpoints/vae_resnet18_baseline/best_vae_model.pth"
    INPUT_FILE = "./dataset/synthetic_raw/synthetic_sino_0015.mrc" 
    OUTPUT_FOLDER = "./dataset/reconstructions/"
    
    TEST_MASK = {'range': (-50, 50), 'step': 5}
    
    run_inference(
        input_mrc_path=INPUT_FILE, 
        checkpoint_path=CHECKPOINT_FILE, 
        output_dir=OUTPUT_FOLDER,
        mask_config=TEST_MASK
    )