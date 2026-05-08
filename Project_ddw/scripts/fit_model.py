# %%
import ast
import inspect
import os
from pathlib import Path
from typing import Optional, List

import pytorch_lightning as pl
import typer
import torch
from typer_config import conf_callback_factory
from typing_extensions import Annotated

from .utils.dataloader import MultiEpochsDataLoader as DataLoader
from .utils.load_function_args_from_yaml_config import load_function_args_from_yaml_config
from .utils.dataset import PatchDataset  # Renamed import
from .utils.unet import LitUnet2D        # Renamed import


loader = lambda yaml_config_file: load_function_args_from_yaml_config(
    function=fit_model_2d, yaml_config_file=yaml_config_file
)
callback = conf_callback_factory(loader)


def fit_model_2d(
    unet_params_dict: Annotated[
        str,
        typer.Option(
            callback=ast.literal_eval,
            help=f"Dictionary of parameters for the 2D U-Net model. See {inspect.getfile(LitUnet2D)} for details.",
        ),
    ],
    adam_params_dict: Annotated[
        str,
        typer.Option(
            callback=ast.literal_eval,
            help="Dictionary of parameters for PyTroch's Adam optimizer.",
        ),
    ],
    num_epochs: Annotated[int, typer.Option(help="Number of epochs to fit the model.")],
    batch_size: Annotated[int, typer.Option(help="Batch size for the optimizer.")],
    patch_size: Annotated[
        int, typer.Option(help="Size of the 2D patches used for model fitting.")
    ],
    mw_angle: Annotated[
        float, typer.Option(help="Width of the PHYSICAL missing wedge in degrees.")
    ],
    gpu: Annotated[List[int], typer.Option(help="Which GPU(s) to use for model fitting. Example: gpu=0 uses the first GPU, gpu=[0,1] uses the first two GPUs.")],
    num_workers: Annotated[
        int, typer.Option(help="Number of CPU workers to use for data loading.")
    ],
    artificial_mw_angle: Annotated[
        Optional[float], typer.Option(help="Width of the ARTIFICIAL missing wedge to drop during training. If None, defaults to mw_angle.")
    ] = None,
    patch_dir: Annotated[
        Optional[str],
        typer.Option(help="Path to the directory containing the patches. Defaults to '{project_dir}/patches'."),
    ] = None,
    project_dir: Annotated[
        Optional[str],
        typer.Option(help="If patch_dir or logdir is not provided, defaults use this project_dir."),
    ] = None,
    logdir: Annotated[
        Optional[str],
        typer.Option(help="Path to save model checkpoints and logs. Defaults to '{project_dir}/logs'."),
    ] = None,
    logger: Annotated[
        str, typer.Option(help="Which PyTorch Lightning logger to use ('tensorboard' or 'csv').")
    ] = "tensorboard",
    check_val_every_n_epochs: Annotated[
        int, typer.Option(help="Check validation loss every n epochs.")
    ] = 10,
    update_patch_missing_wedges_every_n_epochs: Annotated[
        int,
        typer.Option(help="After how many epochs to update the missing wedge in the patches."),
    ] = 10,
    save_model_every_n_epochs: Annotated[
        int, typer.Option(help="Save a model checkpoint to logdir every n epochs.")
    ] = 10,
    save_n_models_with_lowest_fitting_loss: Annotated[
        int, typer.Option(help="Save the n models with the lowest fitting loss to logdir."),
    ] = 5,
    save_n_models_with_lowest_val_loss: Annotated[
        int, typer.Option(help="Save the n models with the lowest validation loss to logdir."),
    ] = 5,
    resume_from_checkpoint: Annotated[
        Optional[str], typer.Option(help="Continue model fitting from a checkpoint.")
    ] = None,
    distributed_backend: Annotated[
        str, 
        typer.Option(help="Distributed backend to use when fitting on multiple GPUs ('nccl' or 'gloo').")
    ] = "nccl",
    seed: Annotated[
        Optional[int], typer.Option(help="Seed for reproducibility.")
    ] = None,
    config: Annotated[
        Optional[str],
        typer.Option(
            callback=callback,
            is_eager=True,
            help="Path to a yaml file containing the arguments.",
        ),
    ] = None,
):
    """
    Fit a 2D U-Net model for missing sector reconstruction on 2D patches.
    """
    pl.seed_everything(seed, workers=True)
    
    if patch_dir is None:
        if project_dir is not None:
            patch_dir = f"{project_dir}/patches"
        else:
            raise ValueError("If project_dir is not provided, patch_dir must be provided.")
            
    if logdir is None:
        if project_dir is not None:
            logdir = f"{project_dir}/logs"
        else:
            raise ValueError("If project_dir is not provided, logdir must be provided.")
            
    logdir = Path(logdir)
    
    if not os.path.exists(logdir.parent):
        os.makedirs(logdir.parent)
        
    if logger == "tensorboard":
        logger_obj = pl.loggers.TensorBoardLogger(logdir.parent, name=logdir.name)
    elif logger == "csv":
        logger_obj = pl.loggers.CSVLogger(logdir.parent, name=logdir.name)
    else:
        raise ValueError(f"Logger '{logger}' not recognized.")
        
    logdir_str = f"{logger_obj.save_dir}/{logger_obj.name}/version_{logger_obj.version}"
    print(f"Saving logs and model checkpoints to '{logdir_str}'")

    # Check for validation data (Single-stream directory structure)
    val_data_exists = (
        os.path.exists(f"{patch_dir}/val_patches")
        and len(os.listdir(f"{patch_dir}/val_patches")) > 0
    )
    if not val_data_exists:
        print("Running model fitting without validation, as no validation data was found!")

    if not patch_size % (2 ** unet_params_dict["num_downsample_layers"]) == 0:
        raise ValueError(
            f"patch_size must be divisible by 2^unet_params_dict['num_downsample_layers']."
        )

    # Setup 2D datasets
    fitting_dataset = PatchDataset(
        patch_dir=f"{patch_dir}/fitting_patches",
        crop_patches_to_size=patch_size,
        mw_angle=mw_angle,
        rotate_patches=True,
        deterministic_rotations=False,
        artificial_mw_angle=artificial_mw_angle, # Pass the new argument!
    )
    if val_data_exists:
        val_dataset = PatchDataset(
            patch_dir=f"{patch_dir}/val_patches",
            crop_patches_to_size=patch_size,
            mw_angle=mw_angle,
            rotate_patches=True,
            deterministic_rotations=True,
            artificial_mw_angle=artificial_mw_angle,
        )

    # Setup callbacks
    callbacks = []
    epoch_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{logdir_str}/checkpoints/epoch",
        filename="{epoch}",
        monitor="epoch",
        verbose=True,
        save_top_k=-1,
        every_n_epochs=save_model_every_n_epochs,
        save_on_train_epoch_end=True,
    )
    callbacks.append(epoch_callback)
    
    if save_n_models_with_lowest_fitting_loss > 0:
        fitting_loss_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f"{logdir_str}/checkpoints/fitting_loss",
            filename="{epoch}-{fitting_loss:.5f}",
            monitor="fitting_loss",
            verbose=True,
            save_top_k=save_n_models_with_lowest_fitting_loss,
            save_on_train_epoch_end=True,
        )
        callbacks.append(fitting_loss_callback)
        
    if save_n_models_with_lowest_val_loss > 0 and val_data_exists:
        val_loss_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f"{logdir_str}/checkpoints/val_loss",
            filename="{epoch}-{val_loss:.5f}",
            monitor="val_loss",
            verbose=True,
            save_top_k=save_n_models_with_lowest_val_loss,
        )
        callbacks.append(val_loss_callback)

    # Initialize the 2D model
    lit_unet = LitUnet2D(
        unet_params=unet_params_dict,
        adam_params=adam_params_dict,
        patch_dir=patch_dir,
        update_patch_missing_wedges_every_n_epochs=update_patch_missing_wedges_every_n_epochs,
    )

    devices = [gpu] if isinstance(gpu, int) else gpu
    strategy = pl.strategies.DDPStrategy(
        process_group_backend=distributed_backend, 
        find_unused_parameters=False,
    ) if len(devices) > 1 else None
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        check_val_every_n_epoch=(check_val_every_n_epochs if val_data_exists else num_epochs),
        deterministic=True,
        logger=logger_obj,
        callbacks=callbacks,
        detect_anomaly=True,
    )

    fitting_dataloader = DataLoader(
        dataset=fitting_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    
    if val_data_exists:
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )
    else:
        val_dataloader = None

    if val_data_exists and resume_from_checkpoint is None:
        trainer.validate(lit_unet, val_dataloader)
        
    trainer.fit(
        ckpt_path=resume_from_checkpoint, 
        model=lit_unet,
        train_dataloaders=fitting_dataloader,
        val_dataloaders=val_dataloader,
    )