import math
import torch

# We will need to update this to 2D in fourier.py next!
from .fourier import get_2d_fft_freqs_on_grid


def get_missing_wedge_mask(grid_size, mw_angle, device="cpu"):
    """
    Produces a 2D binary mask with shape 'grid_size', which can be used to zero-out 
    Fourier components that lie inside a missing wedge with width 'mw_angle'.
    """
    return get_rotated_missing_wedge_mask(
        grid_size=grid_size, 
        mw_angle=mw_angle, 
        rot_angle=0.0, 
        device=device
    )


def get_rotated_missing_wedge_mask(grid_size, mw_angle, rot_angle, device="cpu"):
    """
    Produces a 2D missing wedge mask rotated by 'rot_angle' degrees.
    Instead of interpolating an image rotation, we mathematically rotate the 
    normal vectors defining the wedge for an exact, artifact-free Fourier mask.
    """
    grid = get_2d_fft_freqs_on_grid(grid_size=grid_size, device=device)
    
    # Calculate half-angle
    alpha = torch.deg2rad(torch.tensor(float(mw_angle))) / 2
    
    # Make normal vectors of the two lines that bound the 2D missing wedge
    normal_left = torch.tensor([torch.sin(alpha), torch.cos(alpha)], device=device)
    normal_right = torch.tensor([torch.sin(alpha), -torch.cos(alpha)], device=device)
    
    # Create a 2D rotation matrix from the rot_angle
    theta = torch.deg2rad(torch.tensor(float(rot_angle)))
    rot_mat = torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta),  torch.cos(theta)]
    ], device=device)
    
    # Rotate the normal vectors
    normal_left = rot_mat @ normal_left
    normal_right = rot_mat @ normal_right
    
    # Select all points that lie inside the lines bounding the missing wedge
    grid_size = [int(s) for s in grid_size]
    upper_wedge = torch.logical_or(
        grid.inner(normal_left) >= 0, grid.inner(normal_right) >= 0
    ).reshape(list(grid_size))
    
    lower_wedge = torch.logical_or(
        grid.inner(normal_left) <= 0, grid.inner(normal_right) <= 0
    ).reshape(list(grid_size))
    
    mw_mask = torch.logical_and(upper_wedge, lower_wedge).int()
    return mw_mask