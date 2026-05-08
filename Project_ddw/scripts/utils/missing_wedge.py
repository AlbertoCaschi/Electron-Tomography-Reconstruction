import math
import torch

from .fourier import get_2d_fft_freqs_on_grid
from .rotation import rotate_patch  # We will define this 2D function later


def get_missing_wedge_mask(grid_size, mw_angle, device="cpu"):
    """
    Produces a 2D binary mask with shape 'grid_size', which can be used to zero-out 
    Fourier components that lie inside a missing sector (wedge) with width 'mw_angle'.
    """
    # grid has shape (H*W, 2) where coordinates are (y, x)
    grid = get_2d_fft_freqs_on_grid(grid_size=grid_size, device=device)
    
    # Calculate normals for the two lines that bound the missing sector
    alpha = torch.deg2rad(torch.tensor(float(mw_angle))) / 2
    
    # In the 2D slice (y, x), 'y' is the vertical beam direction and 'x' is horizontal.
    # We map the 3D x-z plane directly to our 2D y-x plane.
    normal_left = torch.tensor([torch.cos(alpha), torch.sin(alpha)], device=device)
    normal_right = torch.tensor([torch.cos(alpha), -torch.sin(alpha)], device=device)
    
    grid_size = [int(s) for s in grid_size]
    
    # Select all points that lie above or below both lines bounding the missing sector
    upper_wedge = torch.logical_or(
        grid.inner(normal_left) >= 0, grid.inner(normal_right) >= 0
    ).reshape(list(grid_size))
    
    lower_wedge = torch.logical_or(
        grid.inner(normal_left) <= 0, grid.inner(normal_right) <= 0
    ).reshape(list(grid_size))
    
    # The intersection of these conditions creates the binary bow-tie/sector mask
    mw_mask = torch.logical_and(upper_wedge, lower_wedge).int()
    return mw_mask


def get_rotated_missing_wedge_mask(
    grid_size, mw_angle, rot_angle, device="cpu"
):
    """
    Convenience function that generates a 2D missing wedge mask and rotates it 
    'rot_angle' degrees in-plane.
    """
    grid_size = torch.tensor(grid_size)
    
    # Enlarge grid size such that the rotated mask doesn't get clipped at the corners
    adjusted_grid_size = (torch.ceil(math.sqrt(2) * grid_size) / 2.0) * 2
    mw_mask = get_missing_wedge_mask(
        grid_size=adjusted_grid_size, 
        mw_angle=mw_angle, 
        device=device
    )
    
    # Rotate the 2D mask. 
    # (rot_axis is removed because 2D rotations only have one plane of rotation)
    mw_mask = (
        rotate_patch(
            patch=mw_mask,
            rot_angle=rot_angle,
            output_shape=grid_size,
            order=1,  # Bilinear interpolation is usually preferred for rotating 2D masks
        )
        .float()
        .to(device)
    )
    return mw_mask