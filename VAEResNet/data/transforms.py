import numpy as np
import random

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
    """
    def __init__(self, max_shift=20, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, input_sino, target_sino):
        if random.random() < self.p:
            shift = random.randint(-self.max_shift, self.max_shift)
            
            input_sino = np.roll(input_sino, shift, axis=1)
            target_sino = np.roll(target_sino, shift, axis=1)
            
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
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input_sino, target_sino):
        if random.random() < self.p:
            input_sino = np.copy(input_sino[:, ::-1])
            target_sino = np.copy(target_sino[:, ::-1])
        return input_sino, target_sino


class RandomIntensityScale:
    """
    Simulates variations in electron beam exposure or object thickness.
    """
    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, input_sino, target_sino):
        if random.random() < self.p:
            scale_factor = random.uniform(*self.scale_range)
            input_sino = input_sino * scale_factor
            target_sino = target_sino * scale_factor
            
            input_sino = np.clip(input_sino, 0.0, 1.0)
            target_sino = np.clip(target_sino, 0.0, 1.0)
            
        return input_sino, target_sino


class ThresholdFilter:
    """
    Forces near-zero background noise to absolute zero to prevent the network 
    from wasting capacity reconstructing gray artifacts outside the projection bands.
    """
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def __call__(self, input_sino, target_sino):
        # Apply thresholding
        input_sino[input_sino < self.threshold] = 0.0
        target_sino[target_sino < self.threshold] = 0.0
        
        return input_sino, target_sino