import torch
from matplotlib import pyplot as plt

from .fourier import fft_2d


def plot_patch(patch, domain="image", figsize=(5, 5)):
    """
    Plot a 2D image/patch in either the image or Fourier domain.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Ensure patch is on CPU for matplotlib
    patch = patch.detach().cpu()
    
    if domain == "image":
        ax.imshow(patch, cmap="gray")
        ax.set_title("2D Image")
        
    elif domain == "fourier":
        # Compute 2D Fourier magnitude
        patch_ft = fft_2d(patch).abs()
        
        # Apply a log-transform for better visual contrast (standard for 2D FFTs)
        # We add 1e-6 to avoid log(0)
        patch_ft_log = torch.log(patch_ft + 1e-6)
        
        ax.imshow(patch_ft_log, cmap="gray")
        ax.set_title("2D Fourier Spectrum")
        
    # layout
    ax.axis("off")
    fig.tight_layout()
    
    return fig

# Note: The TensorBoard function you had commented out is actually fully 2D compatible 
# out of the box because it just string-casts the matplotlib canvas. You can uncomment 
# and use it as-is!