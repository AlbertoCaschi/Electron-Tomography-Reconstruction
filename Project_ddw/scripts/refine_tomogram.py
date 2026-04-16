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

from .fit_model import LitUnet2D
from .utils.fourier import apply_fourier_mask_to_tomo
from .utils.load_function_args_from_yaml_config import \
    load_function_args_from_yaml_config
from .utils.missing_wedge import get_missing_wedge_mask
from .utils.mrctools import load_mrc_data, save_mrc_data
from .utils.normalization import get_avg_model_input_mean_and_std
from .utils.subtomos import extract_subtomos, reassemble_subtomos

loader = lambda yaml_config_file: load_function_args_from_yaml_config(
    function=refine_tomogram, yaml_config_file=yaml_config_file
)
callback = conf_callback_factory(loader)


def refine_tomogram(
    tomo0_files: Annotated[
        List[Path],
        typer.Option(
            help="List of paths to 2D tomograms (mrc files) reconstructed from one half of the tilt series or movie frames."
        ),
    ],
    tomo1_files: Annotated[
        List[Path],
        typer.Option(
            help="List of paths to 2D tomograms (mrc files) reconstructed from the other half of the tilt series or movie frames."
        ),
    ],
    model_checkpoint_file: Annotated[
        Path,
        typer.Option(
            help="Path to a model checkpoint file (.ckpt extension). Checkpoints saved during model fitting can be found in the 'logdir' directory specifed for the 'fit-model' command."
        ),
    ],
    subtomo_size: Annotated[
        int,
        typer.Option(
            help="Size of the 2D patches to extract. This should be the same as the subtomo_size used during model fitting."
        ),
    ],
    mw_angle: Annotated[
        int, typer.Option(help="Width of the missing wedge in degrees.")
    ],
    subtomo_overlap: Annotated[
        Optional[int],
        typer.Option(
            help="Overlap between patches. This determines the stride of the sliding window. If 'None', this is set to '1/3 * subtomo_size'."
        ),
    ] = None,
    standardize_full_tomos: Annotated[
        bool,
        typer.Option(
            help="Set to 'True' if and only if 'standardize_full_tomos' was 'True' for 'ddw fit-model'."
        ),
    ] = False,
    recompute_normalization: Annotated[
        bool,
        typer.Option(
            help="Whether to recompute the mean and variance used to normalize the tomo0s and tomo1s. Recommended to be True."
        ),
    ] = True,
    batch_size: Annotated[
        int, typer.Option(help="Batch size for processing patches.")
    ] = 1,
    return_tomos: Annotated[
        bool,
        typer.Option(
            help="Whether to return the refined tomograms as a list of tensors."
        ),
    ] = False,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to save the refined tomograms."
        ),
    ] = None,
    project_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to the project directory."
        ),
    ] = None,
    num_workers: Annotated[
        int,
        typer.Option(
            help="Number of CPU workers to use."
        ),
    ] = 0,
    gpu: Annotated[
        Optional[List[int]],
        typer.Option(
            help="GPU id on which to run the model."
        ),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(
            callback=callback,
            is_eager=True,
            help="Path to a yaml config file.",
        ),
    ] = None,
):
    """
    Use a fitted 2D U-Net to denoise tomograms and to fill in their missing wedge. Typically run after `fit-model`.
    """
    if output_dir is None:
        if project_dir is not None:
            output_dir = f"{project_dir}/refined_tomograms"
        elif project_dir is None and return_tomos is False:
            raise ValueError(
                "If return_tomos is False, output_dir or project_dir must be provided, otherwise the refined tomograms will be lost."
            )
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if return_tomos is False and output_dir is None:
        raise ValueError(
            "If return_tomos is False, output_dir or project_dir must be provided, otherwise the refined tomograms will be lost."
        )
    if return_tomos:
        tomo_ref = []

    if subtomo_overlap is None:
        subtomo_overlap = int(math.ceil(subtomo_size / 3))

    if hasattr(gpu, "__len__"):
        if len(gpu) > 1:
            print(f"WARNING: Currently, only a single GPU is supported in 'ddw refine-tomogram'. You passed gpu={gpu}. Continuing with gpu={gpu[0]}.")
        
    device = "cpu" if gpu is None else f"cuda:{gpu[0]}"
    lightning_model = (
        LitUnet2D.load_from_checkpoint(model_checkpoint_file).to(device).eval()
    )

    with torch.no_grad():
        for t0_file, t1_file in zip(tomo0_files, tomo1_files):
            if recompute_normalization:
                loc, scale = get_avg_model_input_mean_and_std(
                    tomo_file=t0_file,
                    subtomo_size=subtomo_size,
                    subtomo_extraction_strides=2 * [subtomo_size - subtomo_overlap], # Changed to 2D
                    mw_angle=mw_angle,
                    batch_size=batch_size,
                    standardize=standardize_full_tomos,
                    num_workers=num_workers,
                    verbose=True,
                )
            else:
                loc, scale = (
                    lightning_model.unet.normalization_loc.clone().detach().item(),
                    lightning_model.unet.normalization_scale.clone().detach().item(),
                )

            t_ref = _refine_single_tomogram(
                tomo_file=t0_file,
                lightning_model=lightning_model,
                subtomo_size=subtomo_size,
                subtomo_overlap=subtomo_overlap,
                mw_angle=mw_angle,
                normalization_loc=loc,
                normalization_scale=scale,
                num_workers=num_workers,
                batch_size=batch_size,
                pbar_desc="Refining tomo0",
            )
            t_ref += _refine_single_tomogram(
                tomo_file=t1_file,
                lightning_model=lightning_model,
                subtomo_size=subtomo_size,
                subtomo_overlap=subtomo_overlap,
                mw_angle=mw_angle,
                normalization_loc=loc,
                normalization_scale=scale,
                num_workers=num_workers,
                batch_size=batch_size,
                pbar_desc="Refining tomo1",
            )
            t_ref /= 2
            if return_tomos:
                tomo_ref.append(t_ref)
            if output_dir is not None:
                basename0, ext = os.path.splitext(os.path.basename(t0_file))
                basename1, ext = os.path.splitext(os.path.basename(t1_file))
                basename = f"{basename0}+{basename1}"
                outfile = f"{output_dir}/{basename}_refined{ext}"
                print(f"Saving refined tomogram to {outfile}")
                save_mrc_data(t_ref.cpu(), f"{outfile}", save=True)
    if return_tomos:
        return tomo_ref


def _refine_single_tomogram(
    tomo_file,
    lightning_model,
    subtomo_size,
    subtomo_overlap,
    mw_angle,
    normalization_loc,
    normalization_scale,
    num_workers=0,
    batch_size=1,
    pbar_desc="Refining tomogram",
):
    # Added squeeze() to enforce 2D geometry
    tomo = load_mrc_data(tomo_file).float().squeeze() 
    
    # apply missing wedge mask here to be more consistent with data during model fitting
    mw_mask = get_missing_wedge_mask(tomo.shape, mw_angle, device=tomo.device)
    tomo = apply_fourier_mask_to_tomo(tomo, mw_mask)

    tomo = (tomo / tomo.std()) * torch.tensor(normalization_scale).to(tomo.device)
    tomo = tomo - tomo.mean() + torch.tensor(normalization_loc).to(tomo.device)

    subtomos, subtomo_start_coords = extract_subtomos(
        tomo=tomo.cpu(),
        subtomo_size=subtomo_size,
        subtomo_extraction_strides=2 * [subtomo_size - subtomo_overlap], # Changed to 2D
        enlarge_subtomos_for_rotating=False,
        pad_before_subtomo_extraction=True,
    )
    
    subtomos = TensorDataset(torch.stack(subtomos))
    subtomo_loader = DataLoader(
        subtomos,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model_outputs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(subtomo_loader, desc=pbar_desc):
            batch_subtomos = batch[0].to(lightning_model.device)
            model_output = lightning_model(batch_subtomos)
            model_output = model_output.detach().cpu()
            model_outputs.append(model_output)
    model_outputs = list(torch.concat(model_outputs, 0))

    tomo_ref = reassemble_subtomos(
        subtomos=model_outputs,
        subtomo_start_coords=subtomo_start_coords,
        subtomo_overlap=subtomo_overlap,
        crop_to_size=tomo.shape,
    )
    return tomo_ref