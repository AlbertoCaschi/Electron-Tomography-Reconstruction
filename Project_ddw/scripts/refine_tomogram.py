# %%
import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import tqdm
import typer
from torch.utils.data import DataLoader, TensorDataset
from typer_config import conf_callback_factory
from typing_extensions import Annotated

from .fit_model import LitUnet2D # Updated import
from .utils.fourier import apply_fourier_mask_to_patch # Updated import
from .utils.load_function_args_from_yaml_config import load_function_args_from_yaml_config
from .utils.missing_wedge import get_missing_wedge_mask
from .utils.mrctools import load_mrc_data, save_mrc_data
from .utils.normalization import get_avg_model_input_mean_and_std
from .utils.patching import extract_patches, reassemble_patches # Updated import

loader = lambda yaml_config_file: load_function_args_from_yaml_config(
    function=refine_image_2d, yaml_config_file=yaml_config_file
)
callback = conf_callback_factory(loader)


def refine_image_2d(
    reconstruction_files: Annotated[
        List[Path],
        typer.Option(
            help="List of paths to 2D reconstructions (mrc files) from the FULL sparse tilt series."
        ),
    ],
    model_checkpoint_file: Annotated[
        Path,
        typer.Option(help="Path to a model checkpoint file (.ckpt extension)."),
    ],
    patch_size: Annotated[
        int, typer.Option(help="Size of the square 2D patches to extract.")
    ],
    mw_angle: Annotated[
        int, typer.Option(help="Width of the physical missing wedge in degrees.")
    ],
    artificial_mw_angle: Annotated[
        Optional[int], typer.Option(help="Width of the artificial missing wedge used during training. Defaults to mw_angle.")
    ] = None,
    patch_overlap: Annotated[
        Optional[int],
        typer.Option(help="Overlap between patches. Defaults to '1/3 * patch_size'."),
    ] = None,
    standardize_full_reconstructions: Annotated[
        bool, typer.Option(help="Must match what was used during model fitting.")
    ] = False,
    recompute_normalization: Annotated[
        bool, typer.Option(help="Whether to recompute mean/var for the image individually.")
    ] = True,
    batch_size: Annotated[int, typer.Option(help="Batch size for processing.")] = 1,
    return_images: Annotated[
        bool, typer.Option(help="Whether to return the refined images as a list of tensors.")
    ] = False,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Where to save the refined images.")
    ] = None,
    project_dir: Annotated[
        Optional[Path], typer.Option(help="Path to the project directory.")
    ] = None,
    num_workers: Annotated[
        int, typer.Option(help="Number of CPU workers to use.")
    ] = 0,
    gpu: Annotated[
        Optional[List[int]], typer.Option(help="GPU id on which to run the model.")
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(callback=callback, is_eager=True, help="Path to a yaml config file."),
    ] = None,
):
    """
    Use a fitted 2D U-Net to fill in the missing sector of a 2D FBP reconstruction.
    """
    if artificial_mw_angle is None:
        artificial_mw_angle = mw_angle

    if output_dir is None:
        if project_dir is not None:
            output_dir = f"{project_dir}/refined_images"
        elif project_dir is None and return_images is False:
            raise ValueError("Output_dir or project_dir must be provided.")
            
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if return_images:
        images_ref = []

    if patch_overlap is None:
        patch_overlap = int(math.ceil(patch_size / 3))

    if hasattr(gpu, "__len__") and len(gpu) > 1:
        print(f"WARNING: Only a single GPU is supported. Continuing with gpu={gpu[0]}.")
        
    device = "cpu" if gpu is None else f"cuda:{gpu[0]}"
    lightning_model = LitUnet2D.load_from_checkpoint(model_checkpoint_file).to(device).eval()

    with torch.no_grad():
        for recon_file in reconstruction_files:
            if recompute_normalization:
                loc, scale = get_avg_model_input_mean_and_std(
                    reconstruction_file=recon_file,
                    patch_size=patch_size,
                    patch_extraction_strides=2 * [patch_size - patch_overlap], # 2D change!
                    artificial_mw_angle=artificial_mw_angle, # Pass down the artificial drop
                    mw_angle=mw_angle,
                    batch_size=batch_size,
                    standardize=standardize_full_reconstructions,
                    num_workers=num_workers,
                    verbose=True,
                )
            else:
                loc, scale = (
                    lightning_model.unet.normalization_loc.clone().detach().item(),
                    lightning_model.unet.normalization_scale.clone().detach().item(),
                )

            # Refine the single image
            img_ref = _refine_single_image(
                image_file=recon_file,
                lightning_model=lightning_model,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                mw_angle=mw_angle,
                normalization_loc=loc,
                normalization_scale=scale,
                num_workers=num_workers,
                batch_size=batch_size,
                pbar_desc=f"Refining {os.path.basename(recon_file)}",
            )
            
            if return_images:
                images_ref.append(img_ref)
                
            if output_dir is not None:
                basename, ext = os.path.splitext(os.path.basename(recon_file))
                outfile = f"{output_dir}/{basename}_refined{ext}"
                print(f"Saving refined image to {outfile}")
                save_mrc_data(img_ref.cpu(), f"{outfile}", save=True)
                
    if return_images:
        return images_ref


def _refine_single_image(
    image_file,
    lightning_model,
    patch_size,
    patch_overlap,
    mw_angle,
    normalization_loc,
    normalization_scale,
    num_workers=0,
    batch_size=1,
    pbar_desc="Refining image",
):
    image = load_mrc_data(image_file).float()
    
    # Apply the physical missing wedge mask to be consistent with training data[cite: 17]
    mw_mask = get_missing_wedge_mask(image.shape, mw_angle, device=image.device)
    image = apply_fourier_mask_to_patch(image, mw_mask)

    # Normalize[cite: 17]
    image = (image / image.std()) * torch.tensor(normalization_scale).to(image.device)
    image = image - image.mean() + torch.tensor(normalization_loc).to(image.device)

    # Extract 2D patches[cite: 10, 17]
    patches, patch_start_coords = extract_patches(
        image=image.cpu(),
        patch_size=patch_size,
        patch_extraction_strides=2 * [patch_size - patch_overlap], # 2D tuple
        enlarge_patches_for_rotating=False,
        pad_before_patch_extraction=True,
    )
    
    patches = TensorDataset(torch.stack(patches))
    patch_loader = DataLoader(
        patches,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    model_outputs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(patch_loader, desc=pbar_desc):
            batch_patches = batch[0].to(lightning_model.device)
            model_output = lightning_model(batch_patches)
            model_output = model_output.detach().cpu()
            model_outputs.append(model_output)
            
    model_outputs = list(torch.concat(model_outputs, 0))

    # Stitch the predicted patches back together[cite: 10, 17]
    image_ref = reassemble_patches(
        patches=model_outputs,
        patch_start_coords=patch_start_coords,
        patch_overlap=patch_overlap,
        crop_to_size=image.shape,
    )
    
    return image_ref