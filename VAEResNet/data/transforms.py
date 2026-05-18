import numpy as np
import random
import torch

class PairedCompose:
    """
    Composes several transforms together, ensuring they are applied 
    simultaneously to both the input and the target.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_sino, target_sino):
        for t in self.transforms:
            input_sino, target_sino = t(input_sino, target_sino)
        return input_sino, target_sino


class RandomHorizontalShift:
    """
    Simulates a slight horizontal misalignment of the detector array during acquisition.
    We roll the array horizontally and zero out the wraparound edges.
    """
    def __init__(self, max_shift=20, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, input_sino, target_sino):
        if random.random() < self.p:
            shift = random.randint(-self.max_shift, self.max_shift)
            
            # Roll along the detector axis (axis=1)
            input_sino = np.roll(input_sino, shift, axis=1)
            target_sino = np.roll(target_sino, shift, axis=1)
            
            # Mask out the wrapped-around pixels to be physically accurate
            if shift > 0:
                input_sino[:, :shift] = 0.0
                target_sino[:, :shift] = 0.0
            elif shift < 0:
                input_sino[:, shift:] = 0.0
                target_sino[:, shift:] = 0.0
                
        return input_sino, target_sino


class RandomHorizontalFlip:
    """
    Simulates mirroring the 3D physical object along the horizontal axis.
    Flipping the detector array (axis 1) of a sinogram is physically valid.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input_sino, target_sino):
        if random.random() < self.p:
            # ::-1 reverses the array along the specified axis
            input_sino = np.copy(input_sino[:, ::-1])
            target_sino = np.copy(target_sino[:, ::-1])
        return input_sino, target_sino


class RandomIntensityScale:
    """
    Simulates variations in electron beam exposure or object thickness.
    Multiplies the sinogram by a random scalar.
    """
    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, input_sino, target_sino):
        if random.random() < self.p:
            scale_factor = random.uniform(*self.scale_range)
            input_sino = input_sino * scale_factor
            target_sino = target_sino * scale_factor
            
            # Ensure we don't blow past the [0, 1] normalization limits
            input_sino = np.clip(input_sino, 0.0, 1.0)
            target_sino = np.clip(target_sino, 0.0, 1.0)
            
        return input_sino, target_sino


class ToTensor:
    """
    Converts NumPy arrays to PyTorch Tensors and adds the channel dimension (C, H, W)
    required by the ResNet encoder.
    """
    def __call__(self, input_sino, target_sino):
        # Convert to float32 tensors
        input_tensor = torch.from_numpy(input_sino).float()
        target_tensor = torch.from_numpy(target_sino).float()
        
        # Add a channel dimension: (Angles, Detector) -> (1, Angles, Detector)
        # Note: If your dataset.py already does this, you can skip this step,
        # but it is cleaner to keep all array-to-tensor logic in the transform pipeline.
        if input_tensor.ndim == 2:
            input_tensor = input_tensor.unsqueeze(0)
            target_tensor = target_tensor.unsqueeze(0)
            
        return input_tensor, target_tensor