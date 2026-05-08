import tempfile
import torch
import tqdm

from ..prepare_data import prepare_data_2d # Assuming we rename prepare_data
from .dataset import PatchDataset # Updated import


def get_avg_model_input_mean_and_std(
    reconstruction_file, 
    patch_size, 
    patch_extraction_strides, 
    standardize, 
    mw_angle, 
    batch_size, 
    num_workers, 
    batches=None, 
    verbose=False,
    artificial_mw_angle=None # Added to pass down to PatchDataset
):
    """
    Computes the average mean and standard deviation of model-input-type 2D patches 
    (with the artificial missing sector applied). These values normalize patches 
    during model fitting and final refinement. 
    """
    with tempfile.TemporaryDirectory() as patch_dir:
        # Assuming prepare_data is adapted to single-stream 2D
        prepare_data_2d(
            reconstruction_files=[reconstruction_file], # Single stream!
            mask_files=[],
            patch_size=patch_size,
            extract_larger_patches_for_rotating=True,
            patch_extraction_strides=patch_extraction_strides,  
            val_fraction=0.0,
            patch_dir=patch_dir,
            standardize_full_reconstructions=standardize,
            overwrite=True,
            verbose=False,
        )
        
        dataset = PatchDataset(
            patch_dir=f"{patch_dir}/fitting_patches",
            crop_patches_to_size=patch_size,
            mw_angle=mw_angle,
            rotate_patches=True,
            deterministic_rotations=False,
            artificial_mw_angle=artificial_mw_angle
        )
        
        fitting_dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        
        mean, std = get_avg_model_input_mean_and_std_from_dataloader(
            fitting_dataloader, batches=batches, verbose=verbose
        )
        
    return mean, std


def get_avg_model_input_mean_and_std_from_dataloader(dataloader, batches=None, verbose=False):
    """
    Iterates through the dataloader to compute the global mean and std over 2D patches.
    """
    if batches is None:
        batches = 1 * len(dataloader)
    
    means, vars = [], []
    bar = (
        tqdm.tqdm(range(batches), desc="Computing model-input normalization statistics")
        if verbose
        else range(batches)
    )
    
    iter_loader = iter(dataloader)
    for _ in bar:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            batch = next(iter_loader)
            
        # 2D REDUCTION: Compute mean and var over the Height (-2) and Width (-1)
        means.append(batch["model_input"].mean(dim=(-1, -2)))
        vars.append(batch["model_input"].var(dim=(-1, -2)))
        
    # Aggregate over the batches
    mean = torch.concat(means, 0).mean().cpu().item()
    std = torch.concat(vars, 0).mean().sqrt().cpu().item()
    
    return mean, std