import os
import random
import numpy as np
import mrcfile

# Import your tomography operator
from cDDPM_v2.config import CONFIG
from cDDPM_v2.physics.operators import TomographyOperator


def main():

    # Folder containing sinograms
    sinogram_folder = r"./cDDPM_v2/dataset/synthetic_raw"

    # Find all .mrc files
    mrc_files = [
        os.path.join(sinogram_folder, f)
        for f in os.listdir(sinogram_folder)
        if f.lower().endswith(".mrc")
    ]

    if not mrc_files:
        raise FileNotFoundError(f"No .mrc files found in {sinogram_folder}")

    # Select a random file
    selected_file = random.choice(mrc_files)

    print(f"Selected file: {selected_file}")

    # Load sinogram
    with mrcfile.open(selected_file, permissive=True) as mrc:
        sinogram = np.asarray(mrc.data, dtype=np.float32)

    print(f"Sinogram shape: {sinogram.shape}")

    # Initialize tomography operator
    tomo_operator = TomographyOperator(CONFIG)

    angles = np.arange(0, 181, 1)

    # Perform FBP reconstruction
    reconstruction = tomo_operator.filtered_back_project(sinogram, angles)

    # Compute statistics
    min_val = float(np.min(reconstruction))
    max_val = float(np.max(reconstruction))

    print("\nFBP Reconstruction Statistics")
    print("-----------------------------")
    print(f"Shape     : {reconstruction.shape}")
    print(f"Minimum   : {min_val}")
    print(f"Maximum   : {max_val}")

    # Optional extra statistics
    print(f"Mean      : {float(np.mean(reconstruction))}")
    print(f"Std       : {float(np.std(reconstruction))}")


if __name__ == "__main__":
    main()