import mrcfile
import matplotlib.pyplot as plt
import numpy as np

DATA_FOLDER = "./VAEResNet/dataset/synthetic_raw"
SAMPLE_NUM = "0000"

def check_mrc_dimensions_and_plot(path):
    try:
        with mrcfile.open(path, permissive=True) as mrc:
            data = mrc.data

            print(f"File: {path}")
            print(f"Data type: {data.dtype}")
            print(f"Original shape: {data.shape}")
            print(f"Voxel size: {mrc.voxel_size}")
            print(f"Mode (MRC data type): {mrc.header.mode}")

            if data.ndim == 2:
                img = data

            elif data.ndim == 3:
                img = np.squeeze(data)

                print(f"Squeezed shape: {img.shape}")

                if img.ndim == 3:
                    print("Detected tilt series. Plotting first image.")
                    img = img[0]

                elif img.ndim != 2:
                    raise ValueError(f"Cannot plot array with shape {img.shape}")

            else:
                raise ValueError(f"Unsupported number of dimensions: {data.ndim}")

            # Plot
            plt.figure(figsize=(8, 6))
            plt.imshow(img, cmap="gray", aspect="auto")
            plt.title("Extracted 2D Image / Sinogram")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.colorbar(label="Intensity")
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error reading file: {e}")

check_mrc_dimensions_and_plot(f"{DATA_FOLDER}/synthetic_sino_{SAMPLE_NUM}.mrc")

from skimage.transform import iradon

def reconstruct_and_plot_fbp(path, angles_config=[-90, 1, 90]):
    try:
        with mrcfile.open(path, permissive=True) as mrc:
            data = mrc.data

            if data.ndim == 2:
                sino = data
            elif data.ndim == 3:
                sino = np.squeeze(data)
                if sino.ndim == 3:
                    sino = sino[0]
            else:
                raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


            theta = np.arange(angles_config[0], angles_config[2] + angles_config[1], angles_config[1])

            if sino.shape[0] == len(theta):
                sino = sino.T
            elif sino.shape[1] != len(theta):
                print(f"Warning: Sinogram angle dimension does not match calculated theta length ({len(theta)}).")

            print("Performing Filtered Back Projection (FBP)...")
            
            reconstruction = iradon(sino, theta=theta, filter_name='ramp')

            plt.figure(figsize=(8, 6))
            plt.imshow(reconstruction, cmap="gray")
            plt.title("FBP Reconstructed 2D Image")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.colorbar(label="Density / Attenuation")
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error during FBP reconstruction: {e}")

reconstruct_and_plot_fbp(f"{DATA_FOLDER}/synthetic_sino_{SAMPLE_NUM}.mrc")