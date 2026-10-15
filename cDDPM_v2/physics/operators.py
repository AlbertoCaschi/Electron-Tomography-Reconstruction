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

    def forward_project(self, image, angles): # FBP -> Sinogram
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
        sinogram = radon(image, theta=angles, circle=True)
        
        return sinogram

    def apply_missing_wedge_mask(self, sinogram, angles, missing_wedge_range):
        """
        Sets to 0 all the pixels outside the data wedge in the sinogram.
        
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

    def filtered_back_project(self, sinogram, angles): # Sinogram -> FBP
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
        reconstruction = iradon(sinogram, theta=angles, circle=True, filter_name='ramp')
        
        return reconstruction

    def back_project(self, sinogram, angles): 
        """
        Computes the unfiltered inverse Radon transform (Back-Projection).
        Useful for computing exact mathematical transposes (A^T) for gradient steps
        and variance/uncertainty maps without noise amplification from the ramp filter.
        
        Args:
            sinogram (np.ndarray): The sinogram.
            angles (np.ndarray): 1D array of projection angles in degrees.
            
        Returns:
            np.ndarray: The reconstructed 2D BP image without ramp filtering.
        """
        # filter_name=None removes the ramp filter, applying the exact mathematical transpose of the Radon operator.
        reconstruction = iradon(sinogram, theta=angles, circle=True, filter_name=None)
        
        return reconstruction


import torch
import torch.nn.functional as F

class PyTorchTomographyOperator:
    def __init__(self, detector_pixels, device):
        """
        Initializes the pure-PyTorch physics operator.
        """
        self.detector_pixels = detector_pixels
        self.device = device

    def _get_rotation_matrices(self, angles_deg, batch_size):
        """Generates batched affine rotation matrices."""
        # Convert to radians
        angles_rad = angles_deg * (np.pi / 180.0)
        
        cos_a = torch.cos(angles_rad)
        sin_a = torch.sin(angles_rad)
        
        # Shape: (num_angles, 2, 3)
        matrices = torch.zeros((len(angles_deg), 2, 3), device=self.device)
        matrices[:, 0, 0] = cos_a
        matrices[:, 0, 1] = -sin_a
        matrices[:, 1, 0] = sin_a
        matrices[:, 1, 1] = cos_a
        
        # Repeat for the batch size: (B * num_angles, 2, 3)
        matrices = matrices.repeat(batch_size, 1, 1)
        return matrices

    def forward_project(self, image, angles_deg):
        """
        Forward projection (Radon transform) using grid_sample.
        image: (B, C, H, W) tensor
        angles_deg: 1D tensor or array of angles
        """
        B, C, H, W = image.shape
        num_angles = len(angles_deg)
        
        # Ensure angles are a proper tensor on the correct device
        angles_deg = torch.as_tensor(angles_deg, device=self.device, dtype=torch.float32)

        # Expand image to process all angles simultaneously: (B * num_angles, C, H, W)
        image_expanded = image.unsqueeze(1).expand(B, num_angles, C, H, W).reshape(B * num_angles, C, H, W)
        
        matrices = self._get_rotation_matrices(angles_deg, B)
        
        # Generate grid and rotate
        grid = F.affine_grid(matrices, image_expanded.size(), align_corners=False)
        rotated_images = F.grid_sample(image_expanded, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        # Sum along the Y-axis to simulate X-ray attenuation
        sinogram = rotated_images.sum(dim=2) # Shape: (B * num_angles, C, W)
        
        # Reshape to (B, C, W, num_angles) -> W is detector_pixels
        sinogram = sinogram.view(B, num_angles, C, W).permute(0, 2, 3, 1)
        return sinogram

    def back_project(self, sinogram, angles_deg):
        """
        Unfiltered back-projection.
        sinogram: (B, C, W, num_angles) tensor
        
        NOTE: Because grid_sample uses bilinear interpolation, this is an approximation 
        of the inverse Radon transform, NOT the exact mathematical transpose (A^T). 
        For exact transposes in iterative solvers, use the iterative_reconstruction method instead.
        """
        B, C, W, num_angles = sinogram.shape
        H = W # Output is WxW
        
        angles_deg = torch.as_tensor(angles_deg, device=self.device, dtype=torch.float32)

        # Permute and expand sinogram: smear across the Y-axis
        smeared = sinogram.permute(0, 3, 1, 2).unsqueeze(3).expand(B, num_angles, C, H, W).reshape(B * num_angles, C, H, W)
        
        # For back-projection, we rotate in the opposite direction
        matrices = self._get_rotation_matrices(-angles_deg, B)
        
        grid = F.affine_grid(matrices, smeared.size(), align_corners=False)
        rotated_back = F.grid_sample(smeared, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        # Sum across all angles
        rotated_back = rotated_back.view(B, num_angles, C, H, W)
        reconstruction = rotated_back.sum(dim=1)
        
        # Point 2 Fix: Calculate the actual angular step size (d_theta) in radians
        if len(angles_deg) > 1:
            d_theta = torch.abs(angles_deg[1] - angles_deg[0]) * (torch.pi / 180.0)
        else:
            d_theta = torch.pi / num_angles
            
        reconstruction = reconstruction * d_theta
        return reconstruction

    def filtered_back_project(self, sinogram, angles_deg):
        """
        Filtered back-projection (FBP) using a Ram-Lak (ramp) filter.
        """
        B, C, W, num_angles = sinogram.shape
        
        # Point 1 Fix: Pad the detector dimension to avoid circular convolution artifacts
        pad_len = 2 ** int(np.ceil(np.log2(2 * W)))
        pad_size = pad_len - W
        
        padded_sinogram = F.pad(sinogram, (0, 0, 0, pad_size))
        
        # FFT along the padded detector dimension
        sino_fft = torch.fft.fft(padded_sinogram, dim=2)
        
        # Create the Ramp filter in the frequency domain
        freqs = torch.fft.fftfreq(pad_len, device=self.device)
        ramp_filter = torch.abs(freqs).view(1, 1, pad_len, 1)
        
        # Apply filter and IFFT
        filtered_sino_fft = sino_fft * ramp_filter
        filtered_padded = torch.fft.ifft(filtered_sino_fft, dim=2).real
        
        # Crop back to the original detector width W
        filtered_sinogram = filtered_padded[:, :, :W, :]
        
        # Back-project the filtered sinogram
        return self.back_project(filtered_sinogram, angles_deg)
    