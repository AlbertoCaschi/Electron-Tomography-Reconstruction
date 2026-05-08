import math
import numpy as np
import torch


def extract_patches(
    image,
    patch_size,
    patch_extraction_strides=None,
    enlarge_patches_for_rotating=False,
    pad_before_patch_extraction=False,
):
    """
    Extracts 2D patches of size 'patch_size' using a 2D sliding window. 
    The strides are specified by 'patch_extraction_strides' (two integers).
    """
    if enlarge_patches_for_rotating:
        patch_size = ceil_to_even_integer(math.sqrt(2) * patch_size)
        
    if patch_extraction_strides is None:
        patch_extraction_strides = 2 * [patch_size]
        
    if pad_before_patch_extraction:
        # Pad for extraction based on strides
        pad_y = patch_extraction_strides[0] - (
            (image.shape[0] - patch_size) % patch_extraction_strides[0]
        )
        pad_x = patch_extraction_strides[1] - (
            (image.shape[1] - patch_size) % patch_extraction_strides[1]
        )
        
        # PyTorch pad format: (left, right, top, bottom) -> (0, pad_x, 0, pad_y)
        pad = torch.nn.ReflectionPad2d((0, pad_x, 0, pad_y))
        
        # Apply padding (unsqueeze to add batch/channel dims, then remove them)
        image = pad(image.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        
    # Generate starting indices for each 2D patch (y, x)
    patch_start_coords = [
        (i, j)
        for i in range(0, image.shape[0] - patch_size + 1, patch_extraction_strides[0])
        for j in range(0, image.shape[1] - patch_size + 1, patch_extraction_strides[1])
    ]
    
    # Unfold the image into patches
    patches = (
        image.unfold(0, patch_size, patch_extraction_strides[0])
             .unfold(1, patch_size, patch_extraction_strides[1])
    )
    
    # Reshape to (N, patch_size, patch_size)
    patches = patches.reshape(-1, patch_size, patch_size)
    patches = list(patches)
    
    return patches, patch_start_coords


def reassemble_patches(
    patches, patch_start_coords, patch_overlap=None, crop_to_size=None
):
    """
    Stitches patches back into a single 2D image. Overlapping regions are blended.
    """
    patch_size = patches[0].shape[0]
    
    # Calculate the max indices in 2D
    max_idx = [
        max(start_idx[i] + patch_size for start_idx in patch_start_coords)
        for i in range(2)
    ]
    
    if patch_overlap is None:
        patch_weights = torch.ones_like(patches[0])
    else:
        patch_weights = get_linear_ramp_weights(
            patch_size, patch_overlap
        ).to(patches[0].device)

    out_image = torch.zeros(max_idx, dtype=torch.float32, device=patches[0].device)
    count_image = torch.zeros_like(out_image)
    
    for patch, start_idx in zip(patches, patch_start_coords):
        end_idx = [start + patch_size for start in start_idx]
        
        # Add the weighted patch into the canvas
        out_image[
            start_idx[0] : end_idx[0],
            start_idx[1] : end_idx[1],
        ] += (patch * patch_weights)
        
        # Add the weights to the count canvas for averaging later
        count_image[
            start_idx[0] : end_idx[0],
            start_idx[1] : end_idx[1],
        ] += patch_weights
        
    # Average the overlapping regions
    out_image /= count_image
    
    # Crop back to the original image size if padding was added
    if crop_to_size is not None:
        out_image = out_image[: crop_to_size[0], : crop_to_size[1]]
        
    return out_image


def get_linear_ramp_weights(patch_size, patch_overlap):
    """
    Produces a 2D matrix containing linear weights used to average overlapping patch boundaries.
    """
    ramp = np.linspace(0, 1, patch_overlap) + 1e-6
    weight_map_1d = np.ones(patch_size)
    weight_map_1d[:patch_overlap] = ramp  # Apply sigmoid ramp at the start
    weight_map_1d[-patch_overlap:] = ramp[::-1]  # and at the end, inverted

    # Create a 2D weight map by extending the 1D weight map
    weight_map_2d = np.ones((patch_size, patch_size))
    for i in range(patch_size):
        for j in range(patch_size):
            weight_map_2d[i, j] = weight_map_1d[i] * weight_map_1d[j]

    return torch.from_numpy(weight_map_2d)


def ceil_to_even_integer(x):
    """
    Produces the smallest even integer i that satisfies i >= x.
    """
    return int(math.ceil(x / 2.0) * 2)

# Note: The commented out random-sampling collision functions were removed to save space, 
# but they can be easily adapted by dropping the 'z' coordinate logic if needed.