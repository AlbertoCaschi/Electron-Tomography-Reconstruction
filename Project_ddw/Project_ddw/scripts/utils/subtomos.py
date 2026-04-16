import math

import numpy as np
import torch


def extract_subtomos(
    tomo,
    subtomo_size,
    subtomo_extraction_strides=None,
    enlarge_subtomos_for_rotating=False,
    pad_before_subtomo_extraction=False,
):
    """
    Extracts 2D patches (sub-tomograms) of size 'subtomo_size' using a 2D sliding window approach. 
    The two strides of the sliding window are specified by 'subtomo_extraction_strides', which must be two integers.
    If 'enlarge_subtomos_for_rotating' is True, patches are extracted with shape sqrt(2)*'subtomo_size', 
    so they can be rotated and cropped to 'subtomo_size' without zero-filling.
    """
    if enlarge_subtomos_for_rotating:
        subtomo_size = ceil_to_even_integer(math.sqrt(2) * subtomo_size)
    if subtomo_extraction_strides is None:
        # Changed to 2 dimensions
        subtomo_extraction_strides = 2 * [subtomo_size]
    if pad_before_subtomo_extraction:
        # pad for subtomo extraction with extraction strides
        pad_x = subtomo_extraction_strides[0] - (
            (tomo.shape[0] - subtomo_size) % subtomo_extraction_strides[0]
        )
        pad_y = subtomo_extraction_strides[1] - (
            (tomo.shape[1] - subtomo_size) % subtomo_extraction_strides[1]
        )
        # Changed to ReflectionPad2d and removed pad_z
        pad = torch.nn.ReflectionPad2d((0, pad_y, 0, pad_x))
        # tomo is 2D (H, W), we need to unsqueeze twice to make it (1, 1, H, W) for the pad layer
        tomo = pad(tomo.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        
    # Generating starting indices for each 2D patch
    subtomo_start_coords = [
        (i, j)
        for i in range(
            0, tomo.shape[0] - subtomo_size + 1, subtomo_extraction_strides[0]
        )
        for j in range(
            0, tomo.shape[1] - subtomo_size + 1, subtomo_extraction_strides[1]
        )
    ]
    
    # Unfolding in 2 dimensions instead of 3
    subtomos = (
        tomo.unfold(0, subtomo_size, subtomo_extraction_strides[0])
        .unfold(1, subtomo_size, subtomo_extraction_strides[1])
    )
    
    # Reshaping to 2D patches
    subtomos = subtomos.reshape(-1, subtomo_size, subtomo_size)
    subtomos = list(subtomos)
    return subtomos, subtomo_start_coords


def reassemble_subtomos(
    subtomos, subtomo_start_coords, subtomo_overlap=None, crop_to_size=None
):
    """
    Basically the inverse of 'extract_subtomos'. For this to work, 'extract_subtomos' must have been called 
    with 'pad_before_subtomo_extraction=True', and 'crop_to_size' must be set to the 2D shape of the tomogram 
    from which the patches were extracted.
    """
    # calculate the max indices in each dimension to infer the shape of the original 2D tomogram
    subtomo_size = subtomos[0].shape[0]
    
    # Changed range(3) to range(2)
    max_idx = [
        max(start_idx[i] + subtomo_size for start_idx in subtomo_start_coords)
        for i in range(2)
    ]
    
    if subtomo_overlap is None:
        subtomo_weights = torch.ones_like(subtomos[0])
    else:
        subtomo_weights = get_linear_ramp_weights(
            subtomos[0].shape[0], subtomo_overlap
        ).to(subtomos[0].device)

    out_vol = torch.zeros(max_idx, dtype=torch.float32, device=subtomos[0].device)
    count_vol = torch.zeros_like(out_vol)
    
    for subtomo, start_idx in zip(subtomos, subtomo_start_coords):
        end_idx = [start + subtomo_size for start in start_idx]
        
        # Sliced in 2 dimensions instead of 3
        out_vol[
            start_idx[0] : end_idx[0],
            start_idx[1] : end_idx[1],
        ] += (
            subtomo * subtomo_weights
        )
        count_vol[
            start_idx[0] : end_idx[0],
            start_idx[1] : end_idx[1],
        ] += subtomo_weights
        
    # average the overlapping regions by dividing the accumulated values by their count
    out_vol /= count_vol
    
    if crop_to_size is not None:
        # Cropping in 2 dimensions instead of 3
        out_vol = out_vol[: crop_to_size[0], : crop_to_size[1]]
    return out_vol


def get_linear_ramp_weights(subtomo_size, subtomo_overlap):
    """
    Produces a 2D matrix containing linear weights used to average overlapping patch parts in 'reassemble_subtomos'.
    """
    ramp = np.linspace(0, 1, subtomo_overlap) + 1e-6
    weight_map_1d = np.ones(subtomo_size)
    weight_map_1d[:subtomo_overlap] = ramp  # Apply sigmoid ramp at the start
    weight_map_1d[-subtomo_overlap:] = ramp[::-1]  # and at the end, inverted

    # Create a 2D weight map by extending the 1D weight map to 2 dimensions
    weight_map_2d = np.ones((subtomo_size, subtomo_size))
    for i in range(subtomo_size):
        for j in range(subtomo_size):
            weight_map_2d[i, j] = weight_map_1d[i] * weight_map_1d[j]

    return torch.from_numpy(weight_map_2d)


def ceil_to_even_integer(x):
    """
    Produces the smallest even integer i that satisfies i >= x.
    """
    return int(math.ceil(x / 2.0) * 2)


# --- 2D ADAPTED COMMENTED CODE ---
#
# def try_to_sample_non_overlapping_subtomo_ids(
#     subtomo_start_coords, subtomo_size, target_sample_size, max_tries=1, verbose=True
# ):
#     n = 0
#     most_non_overlapping_subtomo_ids = []
#     while n < max_tries:
#         non_overlapping_subtomo_ids = try_to_sample_non_overlapping_subtomo_ids_(
#             subtomo_start_coords, subtomo_size, target_sample_size
#         )
#         if len(non_overlapping_subtomo_ids) == target_sample_size:
#             return non_overlapping_subtomo_ids
#         elif len(non_overlapping_subtomo_ids) > len(most_non_overlapping_subtomo_ids):
#             most_non_overlapping_subtomo_ids = non_overlapping_subtomo_ids
#             n += 1
#     if verbose:
#         print(
#             f"Warning: Could not sample {target_sample_size} non-overlapping subtomos. "
#         )
#     return most_non_overlapping_subtomo_ids


# def try_to_sample_non_overlapping_subtomo_ids_(
#     subtomo_start_coords, subtomo_size, target_sample_size
# ):
#     if target_sample_size > len(subtomo_start_coords):
#         raise ValueError("n should be less than or equal to the number of subtomos")

#     candidate_ids = list(range(len(subtomo_start_coords)))
#     non_overlapping_subtomo_ids = []

#     n_rejected = 0
#     while len(non_overlapping_subtomo_ids) < target_sample_size:
#         if len(candidate_ids) == 0:
#             return non_overlapping_subtomo_ids
#         idx = random.choice(candidate_ids)
#         starting_index = subtomo_start_coords[idx]
#         # check if sampled patch overlaps with any of the already selected patches
#         overlap = any(
#             [
#                 check_square_overlap(
#                     starting_index, subtomo_start_coords[idx], subtomo_size
#                 )
#                 for idx in non_overlapping_subtomo_ids
#             ]
#         )
#         if not overlap:
#             non_overlapping_subtomo_ids.append(idx)
#         else:
#             n_rejected += 1
#         # remove the sampled patch from the list of indices to sample from
#         candidate_ids.remove(idx)
#     return non_overlapping_subtomo_ids


# def check_square_overlap(starting_point1, starting_point2, square_size):
#     """
#     Checks if two squares of size 'square_size' whose lower-left vertices are 'starting_point1' and 'starting_point2' overlap.
#     """
#     vertices1 = get_square_vertices(starting_point1, square_size)
#     vertices2 = get_square_vertices(starting_point2, square_size)
#     intersect = check_square_intersection(vertices1, vertices2)
#     return intersect


# def get_square_vertices(starting_point, square_size):
#     """
#     Gets coordinates of the vertices of a square of size 'square_size' whose lower-left vertex is 'starting point'.
#     """
#     vertices = []
#     for k in range(2):
#         for j in range(2):
#             vertex = list(starting_point)
#             vertex[k] += square_size * 1
#             vertex[(k + 1) % 2] += square_size * j
#             vertices.append(vertex)
#     vertices = torch.tensor(vertices)
#     return vertices


# def check_square_intersection(vertices1, vertices2):
#     """
#     Checks if two squares with vertices 'vertices1' and 'vertices2' overlap.
#     """
#     intersect_x = (vertices1.min(0).values[0] < vertices2.max(0).values[0]).all() and (
#         vertices1.max(0).values[0] > vertices2.min(0).values[0]
#     ).all()
#     intersect_y = (vertices1.min(0).values[1] < vertices2.max(0).values[1]).all() and (
#         vertices1.max(0).values[1] > vertices2.min(0).values[1]
#     ).all()
#     intersect = intersect_x and intersect_y
#     return intersect