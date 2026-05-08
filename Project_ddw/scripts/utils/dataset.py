import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset

from .fourier import apply_fourier_mask_to_patch
from .missing_wedge import (get_missing_wedge_mask,
                            get_rotated_missing_wedge_mask)
from .rotation import rotate_patch

BASE_SEED = 888


def safe_load(file_path, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            return torch.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}")
            if attempt == max_retries - 1:
                raise e
            print(f"Error message is: {e}")
            print(f"Retrying in {delay} seconds")
            time.sleep(delay)


class PatchDataset(Dataset):
    """
    A torch dataset which produces the input-target 2D patch pairs used for model fitting. 
    Adapted for single-stream Noisier2Noise (no even/odd split).
    """

    def __init__(
        self,
        patch_dir,
        mw_angle,
        crop_patches_to_size,
        rotate_patches=True,
        deterministic_rotations=False,
        artificial_mw_angle=None, # Added to support dynamic wedge dropping
    ):
        super().__init__()
        self.patch_dir = patch_dir
        self.crop_patches_to_size = crop_patches_to_size
        self.mw_angle = mw_angle
        self.rotate_patches = rotate_patches
        self.deterministic_rotations = deterministic_rotations
        
        # If not specified, the artificial wedge is the same size as the physical wedge
        self.artificial_mw_angle = artificial_mw_angle if artificial_mw_angle is not None else mw_angle

    @property
    def rotate_patches(self):
        return self._rotate_patches

    @rotate_patches.setter
    def rotate_patches(self, rotate_patches):
        if not isinstance(rotate_patches, bool):
            raise ValueError("rotate_patches must be a boolean")
        self._rotate_patches = rotate_patches

    def _sample_rot_angle(self, index):
        """Samples a single 2D in-plane rotation angle."""
        seed = BASE_SEED + index if self.deterministic_rotations else None
        rng = np.random.default_rng(seed)
        # Random angle between 0 and 360 degrees
        rot_angle = torch.tensor(rng.uniform(0, 360.0), dtype=torch.float32)
        return rot_angle

    def __len__(self):
        # We only have one directory of patches now, not subtomo0/subtomo1
        return len(os.listdir(self.patch_dir))

    def __getitem__(self, index):
        # 1. Load the SINGLE patch (No split!)
        patch_file = f"{self.patch_dir}/{index}.pt"
        patch = safe_load(patch_file)
        
        if self.rotate_patches:
            rot_angle = self._sample_rot_angle(index)
            
            # 2. Rotate the single patch
            patch = rotate_patch(
                patch,
                rot_angle=rot_angle,
                output_shape=2 * [self.crop_patches_to_size],
            )
            
            # 3. Create the ARTIFICIAL missing wedge mask (M_drop) to apply to the input
            mw_mask = get_missing_wedge_mask(
                grid_size=2 * [self.crop_patches_to_size],
                mw_angle=self.artificial_mw_angle,
                device=patch.device,
            )
            
            # 4. Track the PHYSICAL missing wedge's new location for the loss function
            rot_mw_mask = get_rotated_missing_wedge_mask(
                grid_size=2 * [self.crop_patches_to_size],
                mw_angle=self.mw_angle,
                rot_angle=rot_angle,
                device=patch.device,
            )
        else:
            mw_mask = get_missing_wedge_mask(
                grid_size=patch.shape,
                mw_angle=self.artificial_mw_angle,
                device=patch.device,
            )
            rot_mw_mask = get_missing_wedge_mask(
                grid_size=patch.shape,
                mw_angle=self.mw_angle,
                device=patch.device,
            )
            rot_angle = torch.tensor(0.0)

        # 5. Apply the artificial mask to create the input
        model_input = apply_fourier_mask_to_patch(patch, mw_mask)
        
        # 6. The target is the unaltered (but rotated) single patch
        model_target = patch 

        item = {
            "model_input": model_input,
            "model_target": model_target,
            "mw_mask": mw_mask,
            "rot_mw_mask": rot_mw_mask,
            "patch0_file": patch_file, # Kept this key to maintain compatibility with LitUnet2D
            "rot_angle": rot_angle,
        }
        return item