import os
import torch
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from models.vae import TomographyVAE
from utils.reconstruction import reconstruct_fbp_single

def apply_missing_wedge_mask(sino, tilt_range, step, base_angles_deg):
    """
    Applies the missing wedge mask to a complete sinogram.
    """
    masked_sino = np.zeros_like(sino)
    min_angle, max_angle = tilt_range
    acquired_angles = np.arange(min_angle, max_angle + 1e-5, step)
    
    for angle in acquired_angles:
        idx = np.argmin(np.abs(base_angles_deg - angle))
        masked_sino[idx, :] = sino[idx, :]
        
    return masked_sino

def run_streamlit_inference(
    model,
    input_mrc_path: str,
    output_image_path: str,
    output_fbp_path: str,
    is_complete: bool = True,
    acquisition_config: dict = None,
    threshold: float = 0.05,
    target_size: tuple = (181, 362),
):
    """
    Runs inference on a single sinogram MRC file using the trained VAE.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running inference on {device}...")

    with mrcfile.open(input_mrc_path, permissive=True) as mrc:
        sino = mrc.data.copy().astype(np.float32)
        if sino.ndim == 3 and sino.shape[0] == 1:
            sino = np.squeeze(sino, axis=0)
        if sino.shape[0] == target_size[1]:
            sino = sino.T
            
    if sino.shape != target_size:
        raise ValueError(f"Sinogram shape {sino.shape} does not match target size {target_size}")

    # normalize sino to [0, 1]
    sino_min, sino_max = sino.min(), sino.max()
    if sino_max - sino_min > 1e-6:
        sino = (sino - sino_min) / (sino_max - sino_min)

    base_angles_deg = np.linspace(-90, 90, target_size[0])

    # Masking (for complete sinograms)
    if is_complete:
        if acquisition_config is None:
            raise ValueError("acquisition_config must be provided if is_complete=True")
        input_sino = apply_missing_wedge_mask(sino, acquisition_config['range'], acquisition_config['step'], base_angles_deg)
    else:
        input_sino = sino.copy() # for incomplete sinos

    # threshold
    input_sino[input_sino < threshold] = 0.0

    input_tensor = torch.from_numpy(input_sino).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_tensor, _, _ = model(input_tensor)
        
    pred_sino = pred_tensor.squeeze().cpu().numpy()

    print("Performing FBP Reconstructions...")
    recon_ground_truth_2d = reconstruct_fbp_single(sino, base_angles_deg, filter_name='ramp')
    recon_input_2d = reconstruct_fbp_single(input_sino, base_angles_deg, filter_name='ramp')
    recon_pred_2d = reconstruct_fbp_single(pred_sino, base_angles_deg, filter_name='ramp')


    os.makedirs(os.path.dirname(output_fbp_path), exist_ok=True)
    im = Image.fromarray(recon_pred_2d)
    im.save(output_fbp_path)


    print("Extracting object mask...")
    
    # from [0, 1] values to [0, 255]
    recon_min, recon_max = recon_pred_2d.min(), recon_pred_2d.max()
    recon_uint8 = (255.0 * (recon_pred_2d - recon_min) / (recon_max - recon_min + 1e-8)).astype(np.uint8)

    # Gaussian Blur to smooth out FBP noise and streaking artifacts
    blurred = cv2.GaussianBlur(recon_uint8, (5, 5), 0)

    # automatically separate object from background
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphological closing to fill small holes inside the object mask
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    mask_output_path = output_image_path.replace('.png', '_mask.png')
    os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
    cv2.imwrite(mask_output_path, clean_mask)
    print(f"Binary mask saved to {mask_output_path}")


    print(f"Saving dashboard results to {output_image_path}...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im0 = axes[0, 0].imshow(sino, cmap='gray', aspect='auto')
    axes[0, 0].set_title("Ground Truth")

    im1 = axes[0, 1].imshow(input_sino, cmap='gray', aspect='auto')
    axes[0, 1].set_title("Input Sinogram")
    
    im2 = axes[0, 2].imshow(pred_sino, cmap='gray', aspect='auto')
    axes[0, 2].set_title("VAE Prediction")
    
    im3 = axes[1, 0].imshow(recon_ground_truth_2d, cmap='gray', aspect='equal')
    axes[1, 0].set_title("Ground Truth FBP")
    axes[1, 0].axis('off')

    im4 = axes[1, 1].imshow(recon_input_2d, cmap='gray', aspect='equal')
    axes[1, 1].set_title("FBP from Input")
    axes[1, 1].axis('off')
    
    im5 = axes[1, 2].imshow(recon_pred_2d, cmap='gray', aspect='equal')
    axes[1, 2].set_title("FBP from VAE Prediction")
    axes[1, 2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print("Inference complete.")

def run_inference(
    model_path: str,
    input_mrc_path: str,
    output_image_path: str,
    is_complete: bool = True,
    acquisition_config: dict = None,
    threshold: float = 0.05,
    target_size: tuple = (181, 362),
    latent_dim: int = 64,
    resnet_type: str = 'resnet18'
):
    """
    Runs inference on a single sinogram MRC file using the trained VAE.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running inference on {device}...")

    model = TomographyVAE(
        latent_dim=latent_dim,
        target_size=target_size,
        resnet_type=resnet_type,
        freeze_early_layers=False
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with mrcfile.open(input_mrc_path, permissive=True) as mrc:
        sino = mrc.data.copy().astype(np.float32)
        if sino.ndim == 3 and sino.shape[0] == 1:
            sino = np.squeeze(sino, axis=0)
        if sino.shape[0] == target_size[1]:
            sino = sino.T
            
    if sino.shape != target_size:
        raise ValueError(f"Sinogram shape {sino.shape} does not match target size {target_size}")

    # normalize sino to [0, 1]
    sino_min, sino_max = sino.min(), sino.max()
    if sino_max - sino_min > 1e-6:
        sino = (sino - sino_min) / (sino_max - sino_min)

    base_angles_deg = np.linspace(-90, 90, target_size[0])

    # Masking (for complete sinograms)
    if is_complete:
        if acquisition_config is None:
            raise ValueError("acquisition_config must be provided if is_complete=True")
        input_sino = apply_missing_wedge_mask(sino, acquisition_config['range'], acquisition_config['step'], base_angles_deg)
    else:
        input_sino = sino.copy() # for incomplete sinos

    # threshold
    input_sino[input_sino < threshold] = 0.0

    input_tensor = torch.from_numpy(input_sino).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_tensor, _, _ = model(input_tensor)
        
    pred_sino = pred_tensor.squeeze().cpu().numpy()

    print("Performing FBP Reconstructions...")
    recon_ground_truth_2d = reconstruct_fbp_single(sino, base_angles_deg, filter_name='ramp')
    recon_input_2d = reconstruct_fbp_single(input_sino, base_angles_deg, filter_name='ramp')
    recon_pred_2d = reconstruct_fbp_single(pred_sino, base_angles_deg, filter_name='ramp')


    print("Extracting object mask...")
    
    # from [0, 1] values to [0, 255]
    recon_min, recon_max = recon_pred_2d.min(), recon_pred_2d.max()
    recon_uint8 = (255.0 * (recon_pred_2d - recon_min) / (recon_max - recon_min + 1e-8)).astype(np.uint8)

    # Gaussian Blur to smooth out FBP noise and streaking artifacts
    blurred = cv2.GaussianBlur(recon_uint8, (5, 5), 0)

    # automatically separate object from background
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphological closing to fill small holes inside the object mask
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    mask_output_path = output_image_path.replace('.png', '_mask.png')
    os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
    cv2.imwrite(mask_output_path, clean_mask)
    print(f"Binary mask saved to {mask_output_path}")


    print(f"Saving dashboard results to {output_image_path}...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im0 = axes[0, 0].imshow(sino, cmap='gray', aspect='auto')
    axes[0, 0].set_title("Ground Truth")

    im1 = axes[0, 1].imshow(input_sino, cmap='gray', aspect='auto')
    axes[0, 1].set_title("Input Sinogram")
    
    im2 = axes[0, 2].imshow(pred_sino, cmap='gray', aspect='auto')
    axes[0, 2].set_title("VAE Prediction")
    
    im3 = axes[1, 0].imshow(recon_ground_truth_2d, cmap='gray', aspect='equal')
    axes[1, 0].set_title("Ground Truth FBP")
    axes[1, 0].axis('off')

    im4 = axes[1, 1].imshow(recon_input_2d, cmap='gray', aspect='equal')
    axes[1, 1].set_title("FBP from Input")
    axes[1, 1].axis('off')
    
    im5 = axes[1, 2].imshow(recon_pred_2d, cmap='gray', aspect='equal')
    axes[1, 2].set_title("FBP from VAE Prediction")
    axes[1, 2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Inference complete.")

if __name__ == "__main__":
    
    run_inference(
        model_path="./VAEResNet/checkpoints/vae_resnet18_baseline/train_final.pth",
        input_mrc_path="./VAEResNet/dataset/synthetic_raw/synthetic_sino_0030.mrc", # forma curva semplice
        # input_mrc_path="./VAEResNet/dataset/synthetic_raw/synthetic_sino_0033.mrc", # cubo + forma curva
        # input_mrc_path="./VAEResNet/dataset/synthetic_raw/synthetic_sino_0299.mrc", # due cubi overlap
        # input_mrc_path="./VAEResNet/dataset/synthetic_raw/synthetic_sino_0015.mrc", # cubo + "satelliti"
        # input_mrc_path="./VAEResNet/dataset/synthetic_raw/synthetic_sino_0067.mrc", # forma curva + "satelliti"
        # input_mrc_path="./VAEResNet/dataset/test_data/2_squares.mrc",
        # input_mrc_path="./VAEResNet/dataset/test_data/catalyst.mrc",
        # input_mrc_path="./VAEResNet/dataset/test_data/circle.mrc",
        output_image_path="./VAEResNet/dataset/reconstructions/result.png",
        is_complete=True,   # is_complete must be set to true if the input sinogram is complete (needs to be masked in the specified configuration)
        acquisition_config={'range': (-50, 50), 'step': 5},
        threshold=0.05
    )