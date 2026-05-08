import math
import torch
import numpy as np
from scipy import ndimage


def rotate_patch(patch, rot_angle, output_shape=None, order=3):
    """
    Rotates the 2D tensor 'patch' by 'rot_angle' degrees in-plane. 
    The rotated tensor is center-cropped to 'output_shape'. 
    If 'output_shape' is None, the rotated tensor is cropped to the dimensions of 'patch'.
    """
    # Grab the last two dimensions (H, W)
    patch_shape = torch.tensor(patch.shape[-2:])
    
    if output_shape is None:
        output_shape = patch_shape
        
    # need later for cropping
    crop_offset = [math.floor((ps - cs) / 2) for ps, cs in zip(patch_shape, output_shape)]
    
    if rot_angle != 0:
        if not torch.is_tensor(rot_angle):
            rot_angle = torch.tensor(rot_angle)
            
        rot_angle_rad = torch.deg2rad(rot_angle.float())
        
        # 1. Create a 2x2 rotation matrix for the 2D plane
        cos_a = torch.cos(rot_angle_rad).item()
        sin_a = torch.sin(rot_angle_rad).item()
        rot_mat = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        # 2. Determine offset to rotate around center of the patch
        # -1 because indexing starts at 0
        c_in = 0.5 * (patch_shape - torch.ones(2)).float().numpy()
        offset = c_in - rot_mat @ c_in
        
        # 3. Apply the rotation using affine_transform
        patch = torch.tensor(
            ndimage.affine_transform(
                patch.cpu().numpy(), matrix=rot_mat, offset=offset, order=order
            ),
            device=patch.device,
            dtype=patch.dtype,
        )
        
    # Crop the 2D patch to the desired output shape
    patch = patch[
        ...,
        crop_offset[0] : crop_offset[0] + int(output_shape[0]),
        crop_offset[1] : crop_offset[1] + int(output_shape[1]),
    ]
    return patch