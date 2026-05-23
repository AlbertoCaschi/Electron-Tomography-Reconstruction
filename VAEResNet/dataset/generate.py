import os
import numpy as np
import mrcfile
from skimage.transform import radon
from skimage.draw import disk, polygon
from scipy.interpolate import splprep, splev
from tqdm import tqdm

def create_phantom(size=362):
    """
    Creates a 2D phantom with 1 or 2 specific objects:
    rotated rectangles or smooth irregular blobs.
    """
    image = np.zeros((size, size), dtype=np.float32)
    
    # limit to 1 or 2 objects per image
    num_objects = np.random.randint(1, 3) 
    
    for _ in range(num_objects):
        # choose randomly the type of shape to draw
        shape_type = np.random.choice(['rotated_rect', 'blob'])
        
        # random center of the object in the image
        center_r = np.random.randint(size//4, size*3//4)
        center_c = np.random.randint(size//4, size*3//4)
        intensity = np.random.uniform(0.4, 1.0)
        
        if shape_type == 'rotated_rect':
            # --- Generate a Rotated Rectangle/Square ---
            width = np.random.randint(size//6, size//3)
            height = np.random.randint(size//6, size//3)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Calculate corners using a rotation matrix
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dr = np.array([-height/2, height/2, height/2, -height/2])
            dc = np.array([-width/2, -width/2, width/2, width/2])
            
            rr_coords = center_r + dr * cos_a - dc * sin_a
            cc_coords = center_c + dr * sin_a + dc * cos_a
            
            # Draw the polygon
            rr, cc = polygon(rr_coords, cc_coords, shape=(size, size))
            image[rr, cc] = intensity
            
            # Optional: Sometimes add small "satellite" circles like in your 2nd image
            if np.random.rand() > 0.5:
                num_satellites = np.random.randint(1, 4)
                for _ in range(num_satellites):
                    # Place near the rectangle
                    sat_r = center_r + np.random.randint(-height, height)
                    sat_c = center_c + np.random.randint(-width, width)
                    sat_radius = np.random.randint(8, 20)
                    sat_intensity = np.random.uniform(0.5, 1.0)
                    
                    rr_sat, cc_sat = disk((sat_r, sat_c), sat_radius, shape=(size, size))
                    image[rr_sat, cc_sat] = sat_intensity

        elif shape_type == 'blob':
            # --- Generate a Smooth Irregular Blob ---
            base_radius = np.random.randint(size//6, size//3)
            num_points = 8  # Base vertices
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            
            # Add random noise to radii to make it irregular
            radii = base_radius * np.random.uniform(0.75, 1.25, len(angles))
            
            r = center_r + radii * np.sin(angles)
            c = center_c + radii * np.cos(angles)
            
            # Close the loop
            r = np.append(r, r[0])
            c = np.append(c, c[0])
            
            # Smooth the rough polygon using B-splines
            tck, u = splprep([r, c], s=0, per=True)
            unew = np.linspace(0, 1, 200) # 200 points for a very smooth edge
            out = splev(unew, tck)
            
            # Draw the smoothed polygon
            rr, cc = polygon(out[0], out[1], shape=(size, size))
            image[rr, cc] = intensity

    # Normalize to [0, 1]
    image = np.clip(image, 0.0, 1.0)
    return image

def generate_dataset(output_dir="./VAEResNet/dataset/synthetic_raw", num_samples=10, target_size=(181, 362)):
    """
    Generates phantoms, performs forward projection, and saves as MRC.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_samples} synthetic sinograms...")
    
    num_angles = target_size[0]
    angles_deg = np.linspace(-90, 90, num_angles)
    detector_size = target_size[1]
    
    for i in tqdm(range(num_samples)):
        phantom = create_phantom(size=detector_size)
        sinogram = radon(phantom, theta=angles_deg, circle=True)
        sinogram = sinogram.astype(np.float32)
        
        sino_min, sino_max = sinogram.min(), sinogram.max()
        if sino_max - sino_min > 1e-6:
            sinogram = (sinogram - sino_min) / (sino_max - sino_min)
            
        file_path = os.path.join(output_dir, f"synthetic_sino_{i:04d}.mrc")
        with mrcfile.new(file_path, overwrite=True) as mrc:
            mrc.set_data(sinogram)

if __name__ == "__main__":
    generate_dataset(num_samples=1500)
    print("Synthetic dataset generation complete!")