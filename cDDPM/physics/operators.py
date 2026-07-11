import numpy as np
from skimage.transform import radon, iradon

class TomographyOperator:
    def __init__(self, config):
        """
        Initializes the physics operator for 2D tomographic projections.
        
        Args:
            config (dict): The physics configuration dictionary from config.py.
        """
        self.detector_pixels = config.get("detector_pixels", 362)
        self.geometry_type = config.get("geometry_type", "parallel")
        self.backend = config.get("backend", "skimage")
        
        if self.backend != "skimage":
            print(f"Warning: Backend '{self.backend}' requested, but falling back to "
                  f"'skimage' for safe PyTorch DataLoader multiprocessing.")

    def forward_project(self, image, angles):
        """
        Computes the Radon transform (forward projection) of a 2D image.
        
        Args:
            image (np.ndarray): The clean 2D spatial slice of shape (H, W).
            angles (np.ndarray): 1D array of projection angles in degrees.
            
        Returns:
            np.ndarray: The resulting sinogram of shape (detector_pixels, len(angles)).
        """
        # skimage radon treats the center of the image as the origin of rotation.
        # circle=False ensures the entire image is projected, not just the inscribed circle.
        sinogram = radon(image, theta=angles, circle=False)
        
        return sinogram

    def apply_missing_wedge_mask(self, sinogram, angles, missing_wedge_range):
        """
        Simulates a limited-angle scenario by deterministically masking out specific
        angular ranges in the sinogram.
        
        Args:
            sinogram (np.ndarray): The complete sinogram.
            angles (np.ndarray): 1D array of projection angles in degrees.
            missing_wedge_range (tuple): (min_angle, max_angle) defining the wedge.
            
        Returns:
            np.ndarray: The masked sinogram where the missing wedge region is set to 0.
        """
        masked_sinogram = sinogram.copy()
        mw_min, mw_max = missing_wedge_range
        
        # In electron tomography, missing wedges are typically symmetric around the y-axis
        # (e.g., masking out extreme high tilts on both positive and negative sides).
        # We apply the mask symmetrically using absolute values.
        mask_indices = (np.abs(angles) >= mw_min) & (np.abs(angles) <= mw_max)
        
        # Zero out the columns in the sinogram corresponding to the missing wedge angles
        masked_sinogram[:, mask_indices] = 0.0
        
        return masked_sinogram

    def filtered_back_project(self, sinogram, angles):
        """
        Computes the inverse Radon transform (Filtered Back-Projection) to reconstruct
        the 2D spatial slice from a sinogram.
        
        Args:
            sinogram (np.ndarray): The (potentially masked) sinogram.
            angles (np.ndarray): 1D array of projection angles in degrees.
            
        Returns:
            np.ndarray: The reconstructed 2D FBP image.
        """
        # iradon requires the exact angles used during the forward projection.
        # We use a standard 'ramp' (Ram-Lak) filter which is standard for analytical FBP.
        reconstruction = iradon(sinogram, theta=angles, circle=False, filter_name='ramp')
        
        return reconstruction