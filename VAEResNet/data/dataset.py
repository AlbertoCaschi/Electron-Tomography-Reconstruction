import torch
from torch.utils.data import Dataset
import numpy as np

class TomographyDataset(Dataset):
    """
    PyTorch Dataset for Electron Tomography Sinogram Inpainting.
    Generates masked and noisy input sinograms from complete ground-truth sinograms.
    """
    def __init__(self, ground_truth_sinos, configs, base_angles_deg=None, 
                 is_training=True, transform=None):
        
        self.gt_sinos = ground_truth_sinos
        self.configs = configs
        self.is_training = is_training
        self.transform = transform
        
        if base_angles_deg is None:
            N_angles = self.gt_sinos[0].shape[0]
            self.base_angles_deg = np.linspace(-90, 90, N_angles)
        else:
            self.base_angles_deg = base_angles_deg

        self.samples = []
        for obj_idx in range(len(ground_truth_sinos)):
            for cfg_idx in range(len(configs)):
                self.samples.append((obj_idx, cfg_idx))

    def __len__(self):
        multiplier = 10 if self.is_training else 1
        return len(self.samples) * multiplier

    def __getitem__(self, idx):
        actual_idx = idx % len(self.samples)
        obj_idx, cfg_idx = self.samples[actual_idx]
        
        target_sino = np.copy(self.gt_sinos[obj_idx])
        config = self.configs[cfg_idx]
        
        target_sino = self._normalize(target_sino)

        input_sino = self._apply_missing_wedge_mask(target_sino, config['range'], config['step'])
        
        input_sino = self._add_noise(input_sino)

        if self.transform:
            input_sino, target_sino = self.transform(input_sino, target_sino)

        input_tensor = torch.from_numpy(input_sino).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_sino).float().unsqueeze(0)

        return input_tensor, target_tensor

    def _apply_missing_wedge_mask(self, sino, tilt_range, step):
        masked_sino = np.zeros_like(sino)
        min_angle, max_angle = tilt_range
        acquired_angles = np.arange(min_angle, max_angle + 1e-5, step)
        
        for angle in acquired_angles:
            idx = np.argmin(np.abs(self.base_angles_deg - angle))
            masked_sino[idx, :] = sino[idx, :]
            
        return masked_sino

    def _add_noise(self, sino):
        gaussian_noise = np.random.normal(loc=0.0, scale=0.02, size=sino.shape)
        intensity_noise = np.random.normal(loc=0.0, scale=0.05, size=sino.shape) * sino
        noisy_sino = sino + gaussian_noise + intensity_noise
        return np.clip(noisy_sino, 0.0, 1.0)

    def _normalize(self, sino):
        min_val = sino.min()
        max_val = sino.max()
        if max_val - min_val > 1e-6:
            return (sino - min_val) / (max_val - min_val)
        return sino