import os
import glob
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import mrcfile

from cDDPM_v2.physics.operators import TomographyOperator
from cDDPM_v2.config import CONFIG


class TomographyDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.image_dims = config["data"]["image_dims"]
        self.acq_cfg = config["acquisition"]
        self.views_per_object = self.acq_cfg.get("views_per_object", 5)
        
        # Directories
        data_dir = config["data"]["dataset_path"]
        self.sirt_dir = config["data"].get("sirt_dataset_path", "./cDDPM_v2/dataset/SIRT_dataset")
        csv_path = config["data"].get("csv_path", "./cDDPM_v2/dataset/sirt_configurations.csv")
        
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.mrc")))
        
        if len(all_files) == 0:
            raise FileNotFoundError(f"No .mrc files found in {data_dir}. Please check your path.")
            
        # Load configurations from CSV
        all_configs = []
        with open(csv_path, mode='r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                all_configs.append(row[1:]) # Store the 5 config strings
                
        if len(all_files) != len(all_configs):
            raise ValueError("Mismatch between number of raw MRC files and CSV configuration rows.")
            
        # Train/validation split
        train_samples = config["data"].get("train_samples", 2000)
        
        if self.mode == "train":
            self.file_paths = all_files[:train_samples]
            self.configs = all_configs[:train_samples]
        elif self.mode == "val":
            self.file_paths = all_files[train_samples:]
            self.configs = all_configs[train_samples:]
        else:
            self.file_paths = all_files
            self.configs = all_configs
            
        if len(self.file_paths) == 0:
            raise ValueError(f"No files available for mode '{self.mode}'. Check your dataset folder and split counts.")

        self.physics_operator = TomographyOperator(config["physics"])
        raw_start, raw_end, raw_step = config["physics"]["raw_angles"]
        self.full_angles = np.arange(raw_start, raw_end + raw_step, raw_step)

    def __len__(self):
        return len(self.file_paths) * self.views_per_object

    def _normalize_and_threshold(self, image, threshold=0.0, ref_min=None, ref_max=None):
        img_min = ref_min if ref_min is not None else image.min()
        img_max = ref_max if ref_max is not None else image.max()
        
        if img_max - img_min < 1e-6:
            return np.full_like(image, -1.0)

        img_normalized = (image - img_min) / (img_max - img_min)
        img_normalized = np.clip(img_normalized, 0.0, 1.0)

        if threshold > 0:
            img_normalized = np.where(img_normalized < threshold, 0.0, img_normalized)

        img_scaled = (img_normalized * 2.0) - 1.0
        return img_scaled

    def _apply_spatial_augmentations(self, img_gt, img_cond, img_unc):
        """Applies IDENTICAL 2D flips and 90-degree rotations to GT, Conditioning, and Uncertainty images."""
        if self.mode != "train":
            return img_gt, img_cond, img_unc
            
        if random.random() > 0.5:
            img_gt = np.fliplr(img_gt)
            img_cond = np.fliplr(img_cond)
            img_unc = np.fliplr(img_unc)
            
        if random.random() > 0.5:
            img_gt = np.flipud(img_gt)
            img_cond = np.flipud(img_cond)
            img_unc = np.flipud(img_unc)
            
        return np.ascontiguousarray(img_gt), np.ascontiguousarray(img_cond), np.ascontiguousarray(img_unc)

    def __getitem__(self, idx):
        actual_file_idx = idx // self.views_per_object
        view_idx = idx % self.views_per_object
        
        # 1. Load raw sinogram to compute pristine Ground Truth
        file_path = self.file_paths[actual_file_idx]
        with mrcfile.open(file_path, permissive=True) as mrc:
            raw_sinogram = np.squeeze(mrc.data).astype(np.float32).copy()
            
        if raw_sinogram.shape[0] == len(self.full_angles):
            raw_sinogram = raw_sinogram.T
            
        x_0_np = self.physics_operator.filtered_back_project(raw_sinogram, self.full_angles)
            
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

        H, W = x_0_padded.shape
        if H > target_h or W > target_w:
            start_y = (H - target_h) // 2
            start_x = (W - target_w) // 2
            x_0_padded = x_0_padded[start_y:start_y+target_h, start_x:start_x+target_w]
        
        # 2. Load the corresponding precomputed SIRT image and Uncertainty map
        base_name = os.path.basename(file_path).replace('.mrc', '')
        sirt_filename = f"{base_name}_cfg_{view_idx}.mrc"
        unc_filename = f"{base_name}_cfg_{view_idx}_unc.mrc"
        
        sirt_path = os.path.join(self.sirt_dir, sirt_filename)
        unc_path = os.path.join(self.sirt_dir, unc_filename)
        
        with mrcfile.open(sirt_path, permissive=True) as mrc:
            x_sirt_np = np.squeeze(mrc.data).astype(np.float32).copy()
            
        with mrcfile.open(unc_path, permissive=True) as mrc:
            x_unc_np = np.squeeze(mrc.data).astype(np.float32).copy()
            
        # 3. Apply matched spatial augmentations
        x_0_padded, x_sirt_np, x_unc_np = self._apply_spatial_augmentations(x_0_padded, x_sirt_np, x_unc_np)

        sirt_min = x_sirt_np.min()
        sirt_max = x_sirt_np.max()
        
        # 4. Normalize and threshold
        threshold = self.config["data"]["noise_threshold"]
        x_sirt_processed = self._normalize_and_threshold(
            x_sirt_np, threshold=0.0, ref_min=sirt_min, ref_max=sirt_max
        ) 
        x_0_processed = self._normalize_and_threshold(
            x_0_padded, threshold=threshold, ref_min=sirt_min, ref_max=sirt_max
        )
        
        # Map the [0, 1] uncertainty map to [-1, 1] for U-Net consistency
        x_unc_processed = (np.clip(x_unc_np, 0.0, 1.0) * 2.0) - 1.0
        
        x_0_tensor = torch.from_numpy(x_0_processed).unsqueeze(0)       
        x_sirt_tensor = torch.from_numpy(x_sirt_processed).unsqueeze(0) 
        x_unc_tensor = torch.from_numpy(x_unc_processed).unsqueeze(0)
        
        # 5. Extract geometry configuration from CSV
        cfg_str = self.configs[actual_file_idx][view_idx]
        current_max_tilt, num_projections = map(float, cfg_str.split('_'))
        
        acq_config = torch.tensor([current_max_tilt, num_projections], dtype=torch.float32)
        
        return x_0_tensor, x_sirt_tensor, x_unc_tensor, acq_config