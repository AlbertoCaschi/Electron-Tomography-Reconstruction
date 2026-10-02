import os
import numpy as np
import matplotlib.pyplot as plt
import mrcfile

from cDDPM_v2.config import CONFIG
# Note: Adjust the import if your file is named physics_operators.py instead of operators.py
from cDDPM_v2.physics.operators import TomographyOperator

def visualize_fbp(mrc_file_path):
    if not os.path.exists(mrc_file_path):
        raise FileNotFoundError(f"Sinogram not found at: {mrc_file_path}")

    # Initialize physics configuration and operator
    physics_cfg = CONFIG["physics"]
    physics_operator = TomographyOperator(physics_cfg)

    # Reconstruct the angles array from config
    raw_start, raw_end, raw_step = physics_cfg["raw_angles"]
    raw_angles = np.arange(raw_start, raw_end + raw_step, raw_step)

    # Load the sinogram
    with mrcfile.open(mrc_file_path, permissive=True) as mrc:
        raw_sinogram = np.squeeze(mrc.data).astype(np.float32).copy()

    # Ensure shape is (detector_pixels, num_angles) to prevent skimage errors
    num_angles = len(raw_angles)
    if raw_sinogram.shape[0] == num_angles or raw_sinogram.shape[0] < raw_sinogram.shape[1]:
        raw_sinogram = raw_sinogram.T

    # Failsafe in case the file has a different number of projections than the config expects
    actual_angles = raw_sinogram.shape[1]
    if actual_angles != num_angles:
        raw_angles = np.linspace(raw_start, raw_end, actual_angles)

    # Execute Filtered Back Projection
    fbp_image = physics_operator.filtered_back_project(raw_sinogram, raw_angles)

    # Plot both the Sinogram and the resulting FBP
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(raw_sinogram, cmap='gray', aspect='auto')
    axes[0].set_title(f"Input Sinogram\n({actual_angles} Projections)")
    axes[0].axis('off')
    
    axes[1].imshow(fbp_image, cmap='gray')
    axes[1].set_title("Filtered Back Projection (FBP)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Point this to any .mrc file in your dataset
    TEST_SINO_PATH = "./cDDPM_v2/dataset/synthetic_raw/synthetic_sino_0001.mrc"
    visualize_fbp(TEST_SINO_PATH)