import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import mrcfile

# Importing the physics operator
from cDDPM.physics.operators import TomographyOperator

class TomographyDataset(Dataset):
    def __init__(self, config, mode="train"):
        """
        Initializes the dataset with continuous domain randomization and train/val splitting.
        
        Args:
            config (dict): The global configuration dictionary.
            mode (str): 'train', 'val', or 'test' to handle data splitting.
        """
        self.config = config
        self.mode = mode
        self.image_dims = config["data"]["image_dims"]
        
        # Randomized acquisition settings
        self.acq_cfg = config["acquisition"]
        self.views_per_object = self.acq_cfg.get("views_per_object", 10)
        
        # Gather all .mrc files from the dataset path
        data_dir = config["data"]["dataset_path"]
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.mrc")))
        
        if len(all_files) == 0:
            raise FileNotFoundError(f"No .mrc files found in {data_dir}. Please check your path.")
            
        # Apply the train/validation split
        train_samples = config["data"].get("train_samples", 2500)
        
        if self.mode == "train":
            self.file_paths = all_files[:train_samples]
        elif self.mode == "val":
            # Taking the rest of the dataset for validation
            self.file_paths = all_files[train_samples:]
        else:
            # Fallback for 'test' or inference if pointed to the same directory
            self.file_paths = all_files
            
        if len(self.file_paths) == 0:
            raise ValueError(f"No files available for mode '{self.mode}'. Check your dataset folder and split counts.")
            
        # Initialize the physics operator
        self.physics_operator = TomographyOperator(config["physics"])

    def __len__(self):
        # Artificially expand the dataset length so each object is seen multiple times per epoch
        return len(self.file_paths) * self.views_per_object

    def _normalize_to_ddpm_range(self, image):
        """
        Normalizes an image array to the standard DDPM range of [-1, 1].
        """
        img_min = image.min()
        img_max = image.max()
        
        if img_max - img_min < 1e-6:
            return np.zeros_like(image)
            
        img_normalized = (image - img_min) / (img_max - img_min)
        img_scaled = (img_normalized * 2.0) - 1.0
        
        return img_scaled

    def _get_random_angles(self):
        """
        Dynamically generates a random acquisition geometry for the current sample.
        """
        min_tilt, max_tilt = self.acq_cfg["tilt_bounds"]
        current_max_tilt = random.uniform(min_tilt, max_tilt)
        
        min_proj, max_proj = self.acq_cfg["projection_bounds"]
        num_projections = random.randint(min_proj, max_proj)
        
        # Generate linearly spaced angles (e.g., from -53.2° to +53.2°)
        angles_deg = np.linspace(-current_max_tilt, current_max_tilt, num_projections)
        return angles_deg

    def __getitem__(self, idx):
        """
        Loads the slice, applies random limited-angle forward/back projection, and returns tensors.
        """
        # Determine the actual file to load based on the expanded dataset length
        actual_file_idx = idx // self.views_per_object
        file_path = self.file_paths[actual_file_idx]
        
        # 1. Load the clean 2D spatial slice
        with mrcfile.open(file_path, permissive=True) as mrc:
            x_0_np = np.squeeze(mrc.data).astype(np.float32).copy()
            
        # 2. Pad x_0 to match the target U-Net dimensions (368x368)
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
        
        # 3. Select a randomized acquisition geometry
        angles_deg = self._get_random_angles()
        
        # 4. Physics Simulation Pipeline
        sinogram = self.physics_operator.forward_project(x_0_padded, angles_deg)
        x_fbp_np = self.physics_operator.filtered_back_project(sinogram, angles_deg)
        
        # 5. Normalization to [-1, 1]
        x_0_normalized = self._normalize_to_ddpm_range(x_0_padded)
        x_fbp_normalized = self._normalize_to_ddpm_range(x_fbp_np)
        
        # 6. Tensor Conversion [1, H, W]
        x_0_tensor = torch.from_numpy(x_0_normalized).unsqueeze(0)
        x_fbp_tensor = torch.from_numpy(x_fbp_normalized).unsqueeze(0)
        
        return x_0_tensor, x_fbp_tensor