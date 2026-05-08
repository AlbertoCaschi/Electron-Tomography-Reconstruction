import torch
from torch import fft


def fft_2d(patch, norm="ortho"):
    """
    2D Fourier transform with fftshift.
    Applies the transform over the last two dimensions (Height, Width).
    """
    fft_dim = (-1, -2)
    return fft.fftshift(fft.fft2(patch, dim=fft_dim, norm=norm), dim=fft_dim)


def ifft_2d(patch, norm="ortho"):
    """
    Inverse 2D Fourier transform with fftshift.
    Applies the transform over the last two dimensions (Height, Width).
    """
    fft_dim = (-1, -2)
    return fft.ifft2(fft.ifftshift(patch, dim=fft_dim), dim=fft_dim, norm=norm)


def apply_fourier_mask_to_patch(patch, mask, output="real"):
    """
    Multiplies the 2D Fourier transform of 'patch' with 'mask'. 
    This function is used to add the artificial missing sectors to the model inputs.
    """
    patch_ft = fft_2d(patch)
    patch_ft_masked = patch_ft * mask
    patch_filt = ifft_2d(patch_ft_masked)
    
    if output == "real":
        return patch_filt.real
    elif output == "complex":
        return patch_filt


def get_2d_fft_freqs_on_grid(grid_size, device="cpu"):
    """
    Produces a 2D tensor with shape 'grid_size' whose entries are the spatial 
    frequencies that correspond to the entries of a fourier transform computed with 'fft_2d'.
    """
    # grid_size is now expected to be a tuple/list of length 2: (Height, Width)
    y = torch.fft.fftshift(torch.fft.fftfreq(int(grid_size[0]), device=device))
    x = torch.fft.fftshift(torch.fft.fftfreq(int(grid_size[1]), device=device))
    
    # Creates a 2D Cartesian product of the frequencies
    grid = torch.cartesian_prod(y, x)
    return grid