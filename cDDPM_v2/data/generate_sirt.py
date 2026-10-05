import os
import glob
import csv
import numpy as np
import mrcfile
from skimage.transform import radon, iradon, iradon_sart
from tqdm import tqdm

from cDDPM_v2.config import CONFIG

def generate_limited_sirt_dataset(
    input_dir=CONFIG["data"]["dataset_path"], 
    sirt_output_dir="./cDDPM_v2/dataset/SIRT_dataset",
    csv_path="./cDDPM_v2/dataset/sirt_configurations.csv",
    iterations=CONFIG["physics"]["SIRT_iterations"]
):
    os.makedirs(sirt_output_dir, exist_ok=True)
    
    # 1. Dynamically load the exact angles from config.py
    raw_start, raw_end, raw_step = CONFIG["physics"]["raw_angles"]
    raw_angles = np.arange(raw_start, raw_end + raw_step, raw_step)
    target_h, target_w = CONFIG["data"]["image_dims"]
    
    # 2. Read the configurations from the CSV
    configs = []
    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            configs.append(row[1:])  
            
    mrc_files = sorted(glob.glob(os.path.join(input_dir, "*.mrc")))
    
    if len(mrc_files) != len(configs):
        print(f"Warning: Found {len(mrc_files)} MRC files but {len(configs)} CSV rows.")
    
    # 3. Process each object and its 5 configurations
    for idx, file_path in tqdm(enumerate(mrc_files), total=len(mrc_files), desc="Generating 2500x5 SIRT & Uncertainty files"):
        filename = os.path.basename(file_path)
        base_name = filename.replace('.mrc', '')
        cfgs = configs[idx]
        
        # Load raw sinogram
        with mrcfile.open(file_path, permissive=True) as mrc:
            raw_sinogram = np.squeeze(mrc.data).astype(np.float32).copy()
            
        if raw_sinogram.shape[0] < raw_sinogram.shape[1]:
            raw_sinogram = raw_sinogram.T
            
        num_angles = raw_sinogram.shape[1]
        if len(raw_angles) != num_angles:
            raw_angles = np.linspace(raw_start, raw_end, num_angles)
            
        # Ground Truth padding
        x_0_np = iradon(raw_sinogram, theta=raw_angles, circle=True, filter_name='ramp')
        
        pad_h = max(0, target_h - x_0_np.shape[0])
        pad_w = max(0, target_w - x_0_np.shape[1])
        pad_top, pad_left = pad_h // 2, pad_w // 2
        
        x_0_padded = np.pad(
            x_0_np, 
            ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left)), 
            mode='constant', 
            constant_values=0
        )
        
        # Generate the 5 limited-angle SIRT reconstructions and Uncertainty Maps
        for cfg_idx, cfg_str in enumerate(cfgs):
            max_tilt, num_proj = map(float, cfg_str.split('_'))
            num_proj = int(num_proj)
            
            angles_deg = np.linspace(-max_tilt, max_tilt, num_proj)
            
            # Forward project from the padded GT to get the limited sinogram (pads to ~521)
            limited_sinogram = radon(x_0_padded, theta=angles_deg, circle=True)
            
            # --- A. SIRT (SART) Reconstruction ---
            reconstruction = None
            for _ in range(iterations):
                reconstruction = iradon_sart(limited_sinogram, theta=angles_deg, image=reconstruction)
                
            # --- B. Uncertainty Map Generation ---
            b_maps = []
            for i in range(num_proj):
                angle = np.array([angles_deg[i]])
                sino_slice = limited_sinogram[:, i:i+1]
                # Unfiltered back-projection for exact variance
                b_i = iradon(sino_slice, theta=angle, circle=True, filter_name=None)
                b_maps.append(b_i)
                
            b_maps = np.stack(b_maps, axis=0)
            variance_map = np.var(b_maps, axis=0)
            var_max = (num_proj + 1) / (4 * num_proj)
            u_map = np.clip(variance_map / var_max, 0.0, 1.0)
                
            # --- C. Crop both back to target dimensions ---
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