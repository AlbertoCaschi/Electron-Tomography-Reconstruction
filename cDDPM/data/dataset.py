import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import mrcfile

from cDDPM.physics.operators import TomographyOperator


class TomographyDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.image_dims = config["data"]["image_dims"]
        self.acq_cfg = config["acquisition"]
        self.views_per_object = self.acq_cfg.get("views_per_object", 10)
        data_dir = config["data"]["dataset_path"]
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.mrc")))
        
        if len(all_files) == 0:
            raise FileNotFoundError(f"No .mrc files found in {data_dir}. Please check your path.")
            
        # train/validation split
        train_samples = config["data"].get("train_samples", 2500)
        
        if self.mode == "train":
            self.file_paths = all_files[:train_samples]
        elif self.mode == "val":
            self.file_paths = all_files[train_samples:]
        else:
            self.file_paths = all_files
            
        if len(self.file_paths) == 0:
            raise ValueError(f"No files available for mode '{self.mode}'. Check your dataset folder and split counts.")
            

        self.physics_operator = TomographyOperator(config["physics"])
        
        raw_start, raw_end, raw_step = config["physics"]["raw_angles"]
        self.full_angles = np.arange(raw_start, raw_end + raw_step, raw_step)

    def __len__(self):
        return len(self.file_paths) * self.views_per_object

    def _normalize_to_ddpm_range(self, image):
        img_min = image.min()
        img_max = image.max()
        
        if img_max - img_min < 1e-6:
            return np.zeros_like(image)
            
        img_normalized = (image - img_min) / (img_max - img_min)
        img_scaled = (img_normalized * 2.0) - 1.0
        return img_scaled

    def _get_random_angles(self):
        min_tilt, max_tilt = self.acq_cfg["tilt_bounds"]
        current_max_tilt = random.uniform(min_tilt, max_tilt)
        
        min_proj, max_proj = self.acq_cfg["projection_bounds"]
        num_projections = random.randint(min_proj, max_proj)
        
        angles_deg = np.linspace(-current_max_tilt, current_max_tilt, num_projections)
        return angles_deg

    def __getitem__(self, idx):
        actual_file_idx = idx // self.views_per_object
        file_path = self.file_paths[actual_file_idx]
        
        # load sinogram
        with mrcfile.open(file_path, permissive=True) as mrc:
            raw_sinogram = np.squeeze(mrc.data).astype(np.float32).copy()
            
        # check correct shape (362, 181)
        if raw_sinogram.shape[0] == len(self.full_angles):
            raw_sinogram = raw_sinogram.T
            
        # complete sinogram -> full FBP
        x_0_np = self.physics_operator.filtered_back_project(raw_sinogram, self.full_angles)
            
        # padding to match the target U-Net dimensions (368x368)
        target_h, target_w = self.image_dims
        pad_h = max(0, target_h - x_0_np.shape[0])
        pad_w = max(0, target_w - x_0_np.shape[1])
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        x_0_padded = np.pad(
            x_0_np, 
            ((pad_top, pad_bottom), (pad_left, pad_right)), 
            mode='constant', 
            constant_values=0
        )
        
        # select a randomized acquisition geometry for the missing wedge
        angles_deg = self._get_random_angles()
        
        # simulate the configuration
        limited_sinogram = self.physics_operator.forward_project(x_0_padded, angles_deg)
        x_fbp_np = self.physics_operator.filtered_back_project(limited_sinogram, angles_deg)
        
        # normalize
        x_0_normalized = self._normalize_to_ddpm_range(x_0_padded)
        x_fbp_normalized = self._normalize_to_ddpm_range(x_fbp_np)
        
        # tensor conversion [1, H, W]
        x_0_tensor = torch.from_numpy(x_0_normalized).unsqueeze(0)
        x_fbp_tensor = torch.from_numpy(x_fbp_normalized).unsqueeze(0)
        
        return x_0_tensor, x_fbp_tensor