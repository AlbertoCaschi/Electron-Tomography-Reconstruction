# %%
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional

import torch
import typer
from typer_config import conf_callback_factory
from typing_extensions import Annotated

from .utils.load_function_args_from_yaml_config import (
    load_function_args_from_yaml_config,
)
from .utils.mrctools import load_mrc_data
from .utils.patching import extract_patches  # Renamed import

loader = lambda yaml_config_file: load_function_args_from_yaml_config(
    function=prepare_data_2d, yaml_config_file=yaml_config_file
)
callback = conf_callback_factory(loader)


def prepare_data_2d(
    reconstruction_files: Annotated[
        List[Path],
        typer.Option(
            help="List of paths to 2D reconstructions (mrc files) from the FULL sparse tilt series."
        ),
    ],
    patch_size: Annotated[
        int,
        typer.Option(
            help="Size of the square 2D patches to extract for model fitting. Must be divisible by 2^{num_downsample_layers}."
        ),
    ],
    val_fraction: Annotated[
        float,
        typer.Option(
            help="Fraction of patches to use for validation."
        ),
    ] = 0.1,
    mask_files: Annotated[
        List[Path],
        typer.Option(
            help="List of paths to binary masks (mrc files) outlining the ROI. If none provided, the entire image is used."
        ),
    ] = [],
    min_nonzero_mask_fraction_in_patch: Annotated[
        Optional[float],
        typer.Option(
            help="Minimum fraction of voxels in a patch that correspond to nonzero voxels in the mask."
        ),
    ] = 0.3,
    patch_extraction_strides: Annotated[
        Optional[List[int]],
        typer.Option(
            help="List of 2 integers specifying the 2D strides (y, x) used for extraction. If None, stride 'patch_size' is used."
        ),
    ] = None,
    pad_before_patch_extraction: Annotated[
        bool,
        typer.Option(
            help="Whether to pad the images before extracting patches."
        ),
    ] = False,
    extract_larger_patches_for_rotating: Annotated[
        bool,
        typer.Option(
            help="If True, larger patches of size 'patch_size*sqrt(2)' will be extracted to avoid boundary effects when rotating."
        ),
    ] = True,
    standardize_full_reconstructions: Annotated[
        bool,
        typer.Option(
            help="If 'True', the FBP images will be standardized (mean=0, std=1) before extracting patches."
        ),
    ] = False,
    patch_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to save the patches. If not provided, saves to '{project_dir}/patches'."
        ),
    ] = None,
    project_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="If 'patch_dir' is not provided, defaults to '{project_dir}/patches'."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite the existing patch_dir if it already exists."
        ),
    ] = False,
    seed: Annotated[
        Optional[int],
        typer.Option(help="Controls the randomness of the validation data selection."),
    ] = None,
    verbose: Annotated[bool, typer.Option()] = True,
    config: Annotated[
        Optional[Path],
        typer.Option(
            callback=callback,
            is_eager=True,
            help="Path to a yaml file containing the arguments.",
        ),
    ] = None,
):
    """
    Extract 2D patches from single FBP reconstructions to generate inputs and targets 
    for the single-stream Noisier2Noise model fitting.
    """
    # Create output directories (No subtomo0/1 splits anymore)
    fitting_patch_dir, val_patch_dir = setup_patch_dir(
        patch_dir=patch_dir,
        project_dir=project_dir,
        overwrite=overwrite,
        verbose=verbose,
    )
    
    # Check masks
    if len(mask_files) == 0:
        mask_files = [None] * len(reconstruction_files)
        min_nonzero_mask_fraction_in_patch = 0.0
    else:
        if min_nonzero_mask_fraction_in_patch is None:
            raise ValueError("min_nonzero_mask_fraction_in_patch must be provided if mask_files are provided")
            
    if verbose:
        print(f"Starting 2D patch extraction from {len(reconstruction_files)} image(s).")
        
    fitting_counter, val_counter = 0, 0
    
    for k, (recon_file, mask_file) in enumerate(zip(reconstruction_files, mask_files)):
        image = load_mrc_data(recon_file).float()
        
        if standardize_full_reconstructions:
            if verbose: print(f"Standardizing image '{recon_file}'.")
            image -= image.mean()
            image /= image.std()
        else:
            std = image.std()
            if std < 1e-3:
                print(f"WARNING: Standard deviation of '{recon_file}' is low ({std}). Consider standardization.")
                
        # SINGLE STREAM EXTRACTION
        patches, start_coords = extract_patches(
            image=image,
            patch_size=patch_size,
            patch_extraction_strides=patch_extraction_strides,
            enlarge_patches_for_rotating=extract_larger_patches_for_rotating,
            pad_before_patch_extraction=pad_before_patch_extraction,
        )
        
        if mask_file is not None:
            mask = load_mrc_data(mask_file).float()
        else:
            mask = torch.ones_like(image)
            
        if not (mask == 0).logical_or(mask == 1).all():
            raise ValueError("Mask entries must be either 0 or 1")
            
        patches_mask, _ = extract_patches(
            image=mask,
            patch_size=patch_size,
            patch_extraction_strides=patch_extraction_strides,
            enlarge_patches_for_rotating=extract_larger_patches_for_rotating,
            pad_before_patch_extraction=pad_before_patch_extraction,
        )
        
        selected_patch_ids = [
            i for i, submask in enumerate(patches_mask)
            if (submask.sum() / submask.numel()) >= min_nonzero_mask_fraction_in_patch
        ]
        
        if mask_file is not None and verbose:
            print(f"Masking selected {len(selected_patch_ids)}/{len(patches)} patches extracted from image {k}")
            
        patches = [p for i, p in enumerate(patches) if i in selected_patch_ids]
        start_coords = [c for i, c in enumerate(start_coords) if i in selected_patch_ids]

        num_val_patches = math.ceil(len(patches) * val_fraction)
        val_ids = (
            random.Random(seed).sample(range(len(patches)), num_val_patches)
            if num_val_patches > 0 else []
        )
        fitting_ids = [i for i in range(len(patches)) if i not in val_ids]

        # Save single patches
        for idx in sorted(fitting_ids):
            torch.save(patches[idx].clone(), f"{fitting_patch_dir}/{fitting_counter}.pt")
            fitting_counter += 1

        for idx in sorted(val_ids):
            torch.save(patches[idx].clone(), f"{val_patch_dir}/{val_counter}.pt")
            val_counter += 1

    if verbose:
        print(f"Done with patch extraction.")
        print(f"Saved {fitting_counter} patches for model fitting to '{fitting_patch_dir}'.")
        print(f"Saved {val_counter} patches for validation to '{val_patch_dir}'.")


def setup_patch_dir(patch_dir, project_dir, overwrite, verbose):
    if patch_dir is None:
        if project_dir is not None:
            patch_dir = f"{project_dir}/patches"
        else:
            raise ValueError("patch_dir must be provided if project_dir is not provided")
            
    if verbose:
        print(f"Saving all patches to '{patch_dir}'.")
        
    if os.path.exists(patch_dir):
        if overwrite:
            if verbose: print(f"Removing existing patch directory '{patch_dir}'.")
            shutil.rmtree(patch_dir)
        else:
            raise ValueError(f"patch_dir '{patch_dir}' already exists. Set 'overwrite' to True.")

    fitting_patch_dir = f"{patch_dir}/fitting_patches"
    val_patch_dir = f"{patch_dir}/val_patches"
    
    # We no longer create subtomo0 and subtomo1 subdirectories
    os.makedirs(fitting_patch_dir, exist_ok=False)
    os.makedirs(val_patch_dir, exist_ok=False)
    
    return fitting_patch_dir, val_patch_dir