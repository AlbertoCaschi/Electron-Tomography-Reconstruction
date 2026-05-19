import torch
from torch.utils.data import Dataset
import numpy as np
import random

class TomographyDataset(Dataset):
    """
    PyTorch Dataset for Electron Tomography Sinogram Inpainting.
    Generates masked and noisy input sinograms from complete ground-truth sinograms.
    """
    def __init__(self, ground_truth_sinos, configs, base_angles_deg=None, 
                 is_training=True, transform=None):
        """
        Args:
            ground_truth_sinos (list or np.ndarray): Complete sinograms. 
                                                     Expected shape per sino: (N_angles, N_pixels).
            configs (list of dict): List of acquisition configurations. 
                                    Format: [{'range': (-50, 50), 'step': 5}, ...]
            base_angles_deg (np.ndarray): The angles corresponding to the rows of the ground truth.
                                          Defaults to np.linspace(-90, 90, N_angles).
            is_training (bool): If True, applies random on-the-fly augmentations.
            transform (callable): Optional external torchvision/custom transforms.
        """
        self.gt_sinos = ground_truth_sinos
        self.configs = configs
        self.is_training = is_training
        self.transform = transform
        
        # Determine base angles (assume standard -90 to 90 if not provided)
        if base_angles_deg is None:
            N_angles = self.gt_sinos[0].shape[0]
            self.base_angles_deg = np.linspace(-90, 90, N_angles)
        else:
            self.base_angles_deg = base_angles_deg

        # Flatten the object + config combinations
        # If 3 training objects and 6 configs, self.samples has 18 combinations.
        self.samples = []
        for obj_idx in range(len(ground_truth_sinos)):
            for cfg_idx in range(len(configs)):
                self.samples.append((obj_idx, cfg_idx))

    def __len__(self):
        # We multiply the length during training to ensure longer epochs 
        # since we rely heavily on random on-the-fly augmentations.
        multiplier = 10 if self.is_training else 1
        return len(self.samples) * multiplier

    def __getitem__(self, idx):
        # Handle the multiplier logic
        actual_idx = idx % len(self.samples)
        obj_idx, cfg_idx = self.samples[actual_idx]
        
        # Fetch Ground Truth (Target)
        target_sino = np.copy(self.gt_sinos[obj_idx])
        config = self.configs[cfg_idx]
        
        # Data Augmentation
        if self.is_training:
            target_sino = self._apply_augmentations(target_sino)
            
        # Normalize Target to [0, 1] to keep VAE stable
        target_sino = self._normalize(target_sino)

        # Create Masked Input based on config
        input_sino = self._apply_missing_wedge_mask(target_sino, config['range'], config['step'])
        
        # Add Electron Tomography Noise
        input_sino = self._add_noise(input_sino)

        # Apply any external transforms (if provided)
        if self.transform:
            input_sino, target_sino = self.transform(input_sino, target_sino)

        # Convert to PyTorch tensors and add the Channel dimension [C, H, W] expected by ResNet
        input_tensor = torch.from_numpy(input_sino).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_sino).float().unsqueeze(0)

        return input_tensor, target_tensor

    def _apply_missing_wedge_mask(self, sino, tilt_range, step):
        """
        Zeros out all rows of the sinogram that do not fall exactly on the 
        acquired angles specified by tilt_range and step.
        """
        masked_sino = np.zeros_like(sino)
        
        min_angle, max_angle = tilt_range
        # Acquired angles
        acquired_angles = np.arange(min_angle, max_angle + 1e-5, step)
        
        for angle in acquired_angles:
            # Find the closest row index in the base high-res ground truth
            idx = np.argmin(np.abs(self.base_angles_deg - angle))
            masked_sino[idx, :] = sino[idx, :]
            
        return masked_sino

    def _add_noise(self, sino):
        """
        Simulates Electron Tomography noise.
        Combines intensity-dependent noise (Poisson-like) and sensor noise (Gaussian).
        """
        # Small Gaussian background noise (sensor read noise)
        gaussian_noise = np.random.normal(loc=0.0, scale=0.02, size=sino.shape)
        
        # Intensity dependent noise (brighter regions have higher variance)
        # We simulate this using scaled Gaussian noise since actual Poisson 
        # requires integer counts (which we lose after normalization).
        intensity_noise = np.random.normal(loc=0.0, scale=0.05, size=sino.shape) * sino
        
        noisy_sino = sino + gaussian_noise + intensity_noise
        
        # Re-clip to [0, 1] range
        return np.clip(noisy_sino, 0.0, 1.0)

    def _apply_augmentations(self, sino):
        """
        Physically valid augmentations for tomographic sinograms.
        """
        # Horizontal Shift (Translation along the detector array)
        if random.random() > 0.5:
            shift = random.randint(-20, 20)
            # Use roll to shift, simulating the object moving slightly off-center
            sino = np.roll(sino, shift, axis=1)
            
            # Zero out the wrapped edges to be physically accurate
            if shift > 0:
                sino[:, :shift] = 0
            elif shift < 0:
                sino[:, shift:] = 0
                
        # Global Intensity Scaling (Simulating varying beam exposure)
        if random.random() > 0.5:
            scale_factor = random.uniform(0.8, 1.2)
            sino = sino * scale_factor
            
        return sino

    def _normalize(self, sino):
        """Min-Max normalization per-sinogram."""
        min_val = sino.min()
        max_val = sino.max()
        if max_val - min_val > 1e-6:
            return (sino - min_val) / (max_val - min_val)
        return sino