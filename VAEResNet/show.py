import mrcfile
import matplotlib.pyplot as plt
import numpy as np

DATA_FOLDER = "/Users/albertocaschi/Desktop/GenAI project/datasets"
OBJECT = "2_squares"
ANGLES = [-90, 1, 90]

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


check_mrc_dimensions_and_plot(f"{DATA_FOLDER}/{OBJECT}/sinograms/sinogram_{ANGLES[0]}_{ANGLES[1]}_{ANGLES[2]}.mrc")
