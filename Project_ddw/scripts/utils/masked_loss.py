import torch

from .fourier import apply_fourier_mask_to_patch


def masked_loss(model_output, target, rot_mw_mask, mw_mask, mw_weight=100.0):
    """
    The self-supervised per-sample loss function for missing sector reconstruction.
    Adapted for 2D single-stream patches.
    """
    # 1. Data Consistency (Outside the missing wedge)
    # rot_mw_mask zeros out the artificial drop; mw_mask zeros out the physical drop.
    # This evaluates frequencies present in BOTH the input and the target.
    outside_mw_mask = rot_mw_mask * mw_mask
    outside_mw_loss = (
        apply_fourier_mask_to_patch(
            patch=target - model_output, mask=outside_mw_mask, output="real"
        )
        .abs()
        .pow(2)
        .mean()
    )
    
    # 2. Missing Wedge Recovery (Inside the missing wedge)
    # (1 - mw_mask) isolates the frequencies that are missing in the physical acquisition.
    # We evaluate the network's ability to predict the frequencies we artificially dropped.
    inside_mw_mask = rot_mw_mask * (torch.ones_like(mw_mask) - mw_mask)
    inside_mw_loss = (
        apply_fourier_mask_to_patch(
            patch=target - model_output, mask=inside_mw_mask, output="real"
        )
        .abs()
        .pow(2)
        .mean()
    )
    
    # Combine losses. mw_weight controls how hard the network tries to fill the wedge 
    # relative to maintaining data consistency.
    loss = outside_mw_loss + mw_weight * inside_mw_loss
    
    return loss