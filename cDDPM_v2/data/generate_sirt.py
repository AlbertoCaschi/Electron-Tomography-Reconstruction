import os
import glob
import csv
import torch
import numpy as np
import mrcfile
from skimage.transform import iradon_sart
from tqdm import tqdm

from cDDPM_v2.config import CONFIG
from cDDPM_v2.physics.operators import PyTorchTomographyOperator

def get_device():
    """Returns the optimal available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def generate_limited_sirt_dataset(
    input_dir=CONFIG["data"]["dataset_path"], 
    sirt_output_dir="./cDDPM_v2/dataset/SIRT_dataset",
    csv_path="./cDDPM_v2/dataset/sirt_configurations.csv",
    iterations=CONFIG["physics"]["SIRT_iterations"]
):
    os.makedirs(sirt_output_dir, exist_ok=True)
    
    device = get_device()
    print(f"Generating dataset using device: {device}")
    
    # Initialize the new GPU-native physics operator
    physics_operator = PyTorchTomographyOperator(CONFIG["physics"]["detector_pixels"], device)
    
    raw_start, raw_end, raw_step = CONFIG["physics"]["raw_angles"]
    raw_angles = np.arange(raw_start, raw_end + raw_step, raw_step)
    target_h, target_w = CONFIG["data"]["image_dims"]
    
    configs = []
    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            configs.append(row[1:])  
            
    mrc_files = sorted(glob.glob(os.path.join(input_dir, "*.mrc")))
    
    if len(mrc_files) != len(configs):
        print(f"Warning: Found {len(mrc_files)} MRC files but {len(configs)} CSV rows.")
    
    for idx, file_path in tqdm(enumerate(mrc_files), total=len(mrc_files), desc=f"Generating SIRT & Uncertainty map files"):
        filename = os.path.basename(file_path)
        base_name = filename.replace('.mrc', '')
        cfgs = configs[idx]
        
        # Load raw sinogram
        with mrcfile.open(file_path, permissive=True) as mrc:
            raw_sinogram = np.squeeze(mrc.data).astype(np.float32).copy()
            
        if raw_sinogram.shape[0] == len(raw_angles):
            raw_sinogram = raw_sinogram.T
            
        num_angles = raw_sinogram.shape[1]
        if len(raw_angles) != num_angles:
            raw_angles = np.linspace(raw_start, raw_end, num_angles)
            
        # Convert to Tensor (B, C, W, num_angles) for PyTorch FBP
        raw_sino_tensor = torch.from_numpy(raw_sinogram).unsqueeze(0).unsqueeze(0).to(device)
        
        # Ground Truth FBP natively on GPU
        x_0_tensor = physics_operator.filtered_back_project(raw_sino_tensor, raw_angles)
        
        # Drop back to numpy for padding and thresholding
        x_0_np = x_0_tensor.squeeze().cpu().numpy()
        x_0_np = np.clip(x_0_np, a_min=0.0, a_max=None)
        
        pad_h = max(0, target_h - x_0_np.shape[0])
        pad_w = max(0, target_w - x_0_np.shape[1])
        pad_top, pad_left = pad_h // 2, pad_w // 2
        
        # Padded border now perfectly matches the clamped background
        x_0_padded = np.pad(
            x_0_np, 
            ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left)), 
            mode='constant', 
            constant_values=0.0
        )
        
        threshold = CONFIG["data"]["noise_threshold"]
        x_0_normalized = (x_0_padded - x_0_padded.min()) / (x_0_padded.max() - x_0_padded.min() + 1e-8)
        x_0_standardized = np.where(x_0_normalized < threshold, 0.0, x_0_normalized)
        
        # Convert the clean, thresholded [0, 1] GT back to a GPU Tensor
        x_0_standardized_tensor = torch.from_numpy(x_0_standardized).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        
        # Generate the limited-angle SIRT reconstructions and Uncertainty Maps
        for cfg_idx, cfg_str in enumerate(cfgs):
            max_tilt, num_proj = map(float, cfg_str.split('_'))
            num_proj = int(num_proj)
            
            # Using linspace for absolute number of projections
            angles_deg = np.linspace(-max_tilt, max_tilt, num_proj)
            
            # --- A. Forward Project on GPU ---
            limited_sinogram_tensor = physics_operator.forward_project(x_0_standardized_tensor, angles_deg)
        
            
            # --- B. SIRT (SART) Reconstruction on GPU ---
            reconstruction_tensor = physics_operator.iterative_reconstruction(
                target_sinogram=limited_sinogram_tensor, 
                angles_deg=angles_deg, 
                num_iterations=iterations,
                lr=1e-2 
            )

            # Bring the final reconstructed volume back to numpy for cropping
            reconstruction = reconstruction_tensor.squeeze().cpu().numpy()
                
            # --- C. Uncertainty Map Generation on GPU ---
            b_maps = []
            for i in range(num_proj):
                angle = torch.tensor([angles_deg[i]], device=device, dtype=torch.float32)
                sino_slice = limited_sinogram_tensor[:, :, :, i:i+1]
                
                # Raw unfiltered back-projection
                b_i = physics_operator.back_project(sino_slice, angle)
                b_maps.append(b_i)
                
            # Stack and compute variance entirely on GPU
            b_maps_tensor = torch.cat(b_maps, dim=0) # Shape: (num_proj, 1, H, W)
            variance_map_tensor = torch.var(b_maps_tensor, dim=0, keepdim=True)
            
            # Bring variance map to numpy for normalization and cropping
            variance_map = variance_map_tensor.squeeze().cpu().numpy()
            
            # Dynamically min-max normalize the variance to [0, 1]
            var_min, var_max = variance_map.min(), variance_map.max()
            if var_max - var_min > 1e-8:
                u_map = (variance_map - var_min) / (var_max - var_min)
            else:
                u_map = np.zeros_like(variance_map)
                
            # --- D. Crop both back to target dimensions ---
            H, W = reconstruction.shape
            if H > target_h or W > target_w:
                start_y = (H - target_h) // 2
                start_x = (W - target_w) // 2
                reconstruction = reconstruction[start_y:start_y+target_h, start_x:start_x+target_w]
                u_map = u_map[start_y:start_y+target_h, start_x:start_x+target_w]
            
            # Force contiguous memory
            reconstruction = np.ascontiguousarray(reconstruction, dtype=np.float32)
            u_map = np.ascontiguousarray(u_map, dtype=np.float32)
            
            # Save SIRT
            out_name_sirt = f"{base_name}_cfg_{cfg_idx}.mrc"
            out_path_sirt = os.path.join(sirt_output_dir, out_name_sirt)
            with mrcfile.new(out_path_sirt, overwrite=True) as mrc_out:
                mrc_out.set_data(reconstruction)
                
            # Save Uncertainty Map
            out_name_unc = f"{base_name}_cfg_{cfg_idx}_unc.mrc"
            out_path_unc = os.path.join(sirt_output_dir, out_name_unc)
            with mrcfile.new(out_path_unc, overwrite=True) as mrc_out:
                mrc_out.set_data(u_map)

if __name__ == "__main__":
    generate_limited_sirt_dataset()