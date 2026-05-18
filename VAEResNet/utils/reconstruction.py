import torch
import numpy as np
from skimage.transform import iradon, iradon_sart
import warnings

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Safely detaches a PyTorch tensor and moves it to CPU as a NumPy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def reconstruct_fbp_single(sinogram: np.ndarray, angles_deg: np.ndarray, filter_name: str = 'ramp') -> np.ndarray:
    """
    Reconstructs a single 2D image from a sinogram using Filtered Back Projection (FBP).
    
    Args:
        sinogram (np.ndarray): Shape (num_angles, detector_pixels) or (detector_pixels, num_angles).
        angles_deg (np.ndarray): 1D array of projection angles in degrees.
        filter_name (str): Filter used for FBP (e.g., 'ramp', 'shepp-logan', 'hamming').
        
    Returns:
        np.ndarray: The 2D reconstructed image of shape (detector_pixels, detector_pixels).
    """
    # skimage's iradon expects the shape to be (detector_pixels, num_angles)
    if sinogram.shape[0] == len(angles_deg):
        sinogram = sinogram.T
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Suppress skimage padding warnings
        reconstruction = iradon(sinogram, theta=angles_deg, filter_name=filter_name)
        
    return reconstruction

def reconstruct_sart_single(sinogram: np.ndarray, angles_deg: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Reconstructs a single 2D image using SART. Iterative methods are generally 
    superior for electron tomography data with missing wedges or high noise.
    
    Args:
        sinogram (np.ndarray): Shape (num_angles, detector_pixels).
        angles_deg (np.ndarray): 1D array of projection angles in degrees.
        iterations (int): Number of SART iterations.
        
    Returns:
        np.ndarray: The 2D reconstructed image.
    """
    if sinogram.shape[0] == len(angles_deg):
        sinogram = sinogram.T
        
    reconstruction = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(iterations):
            reconstruction = iradon_sart(sinogram, theta=angles_deg, image=reconstruction)
            
    return reconstruction

def batch_reconstruct(sinograms: torch.Tensor, angles_deg: np.ndarray, method: str = 'fbp', **kwargs) -> torch.Tensor:
    """
    A PyTorch-friendly wrapper to reconstruct a batch of predicted sinograms during 
    the validation loop.
    
    Args:
        sinograms (torch.Tensor): Output from VAE decoder. Expected shape (B, 1, num_angles, detector_pixels).
        angles_deg (np.ndarray): The corresponding angles for the complete sinogram.
        method (str): 'fbp' or 'sart'.
        
    Returns:
        torch.Tensor: Reconstructed batch of shape (B, 1, detector_pixels, detector_pixels) 
                      moved back to the same device as the input tensor.
    """
    device = sinograms.device
    sinograms_np = to_numpy(sinograms)
    
    # Check shape: (B, 1, H, W). If it lacks the channel dim, add it.
    if sinograms_np.ndim == 3:
        sinograms_np = np.expand_dims(sinograms_np, axis=1)
        
    B, C, H, W = sinograms_np.shape
    assert C == 1, "Expected single-channel (grayscale) sinograms."
    
    reconstructions = []
    
    for i in range(B):
        sino_2d = sinograms_np[i, 0, :, :] # Extract (num_angles, detector_pixels)
        
        if method.lower() == 'fbp':
            recon = reconstruct_fbp_single(sino_2d, angles_deg, filter_name=kwargs.get('filter_name', 'ramp'))
        elif method.lower() == 'sart':
            recon = reconstruct_sart_single(sino_2d, angles_deg, iterations=kwargs.get('iterations', 1))
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
            
        reconstructions.append(recon)
        
    # Convert list of arrays back to a PyTorch tensor (B, 1, pixels, pixels)
    reconstructions_np = np.array(reconstructions)[:, np.newaxis, :, :]
    
    return torch.from_numpy(reconstructions_np).float().to(device)