import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import mrcfile

# Assuming the physics operator is implemented as we outlined in the blueprint
from physics.operators import TomographyOperator


class TomographyDataset(Dataset):
    def __init__(self, config, mode="train"):
        """
        Initializes the dataset, loads file paths, and sets up the physics operator.
        
        Args:
            config (dict): The global configuration dictionary.
            mode (str): 'train', 'val', or 'test' to handle data splitting if necessary.
        """
        self.config = config
        self.mode = mode
        self.image_dims = config["data"]["image_dims"]
        self.acquisition_configs = config["acquisition_configs"]
        
        # Gather all .mrc files from the dataset path
        data_dir = config["data"]["dataset_path"]
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.mrc")))
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .mrc files found in {data_dir}. Please check your path.")
            
        # Initialize the physics operator
        # Note: If TomographyOperator uses GPU backends (like ASTRA), be mindful 
        # of PyTorch DataLoader multiprocessing (num_workers > 0) which can cause CUDA context errors.
        self.physics_operator = TomographyOperator(config["physics"])

    def __len__(self):
        return len(self.file_paths)

    def _normalize_to_ddpm_range(self, image):
        """
        Normalizes an image array to the standard DDPM range of [-1, 1].
        Assumes the input image can have arbitrary physical attenuation values.
        """
        img_min = image.min()
        img_max = image.max()
        
        # Avoid division by zero for completely flat images
        if img_max - img_min < 1e-6:
            return np.zeros_like(image)
            
        # Min-max scale to [0, 1]
        img_normalized = (image - img_min) / (img_max - img_min)
        # Scale to [-1, 1]
        img_scaled = (img_normalized * 2.0) - 1.0
        
        return img_scaled

    def _get_random_angles(self):
        """
        Randomly selects an acquisition configuration and generates the corresponding angle array.
        """
        acq_config = random.choice(self.acquisition_configs)
        start_angle, end_angle = acq_config["range"]
        step = acq_config["step"]
        
        # Generate angles in degrees
        angles_deg = np.arange(start_angle, end_angle + step, step)
        return angles_deg

    def __getitem__(self, idx):
        """
        Loads the clean slice, pads it to U-Net friendly dimensions, 
        generates the simulated measurement, applies the mask, 
        and computes the FBP initialization.
        """
        file_path = self.file_paths[idx]
        
        # 1. Load the clean 2D spatial slice
        with mrcfile.open(file_path, permissive=True) as mrc:
            x_0_np = np.squeeze(mrc.data).astype(np.float32).copy()
            
        # --- NEW CODE: Pad x_0 to match the target U-Net dimensions (e.g., 368x368) ---
        target_h, target_w = self.config["data"]["image_dims"]
        
        # Calculate padding amounts
        pad_h = max(0, target_h - x_0_np.shape[0])
        pad_w = max(0, target_w - x_0_np.shape[1])
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad with zeros (background) to center the object in the field of view
        x_0_padded = np.pad(
            x_0_np, 
            ((pad_top, pad_bottom), (pad_left, pad_right)), 
            mode='constant', 
            constant_values=0
        )
        
        # 2. Select acquisition geometry
        angles_deg = self._get_random_angles()
        
        # 3. Physics Simulation Pipeline (Now using the padded square image)
        sinogram = self.physics_operator.forward_project(x_0_padded, angles_deg)
                
        x_fbp_np = self.physics_operator.filtered_back_project(sinogram, angles_deg)
        
        # 4. Normalization to [-1, 1] for DDPM
        x_0_normalized = self._normalize_to_ddpm_range(x_0_padded)
        x_fbp_normalized = self._normalize_to_ddpm_range(x_fbp_np)
        
        # 5. Tensor Conversion [1, H, W]
        x_0_tensor = torch.from_numpy(x_0_normalized).unsqueeze(0)
        x_fbp_tensor = torch.from_numpy(x_fbp_normalized).unsqueeze(0)
        
        return x_0_tensor, x_fbp_tensor