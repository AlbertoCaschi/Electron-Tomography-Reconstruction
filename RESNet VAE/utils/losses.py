import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAELoss(nn.Module):
    """
    Computes the combined loss for the Variational Autoencoder.
    L_total = L_recon + beta * D_KL
    """
    def __init__(self, recon_loss_type='l1'):
        """
        Args:
            recon_loss_type (str): 'l1' (MAE) or 'mse' (L2). L1 is highly recommended 
                                   for tomographic sinograms to preserve sharp edges.
        """
        super().__init__()
        self.recon_loss_type = recon_loss_type.lower()

    def forward(self, recon_x, x, mu, logvar, beta=1.0):
        """
        Args:
            recon_x: Predicted sinogram from the decoder (B, C, H, W)
            x: Ground truth complete sinogram (B, C, H, W)
            mu: Latent space mean (B, latent_dim)
            logvar: Latent space log-variance (B, latent_dim)
            beta: Current weight for the KL Divergence (from scheduler)
            
        Returns:
            total_loss (Tensor): The combined loss for backpropagation.
            recon_loss (Tensor): Pure reconstruction loss (for logging).
            kl_div (Tensor): Pure KL divergence (for logging).
        """
        batch_size = x.size(0)

        # 1. Reconstruction Loss
        if self.recon_loss_type == 'l1':
            # reduction='none' allows us to sum over the image dimensions first
            recon_loss = F.l1_loss(recon_x, x, reduction='none')
        elif self.recon_loss_type == 'mse':
            recon_loss = F.mse_loss(recon_x, x, reduction='none')
        else:
            raise ValueError(f"Unknown reconstruction loss type: {self.recon_loss_type}")
            
        # Sum over channels, height, width, then average over the batch
        recon_loss = recon_loss.view(batch_size, -1).sum(dim=1).mean()

        # 2. KL Divergence
        # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean() # Average over the batch

        # 3. Total Beta-VAE Loss
        total_loss = recon_loss + (beta * kl_loss)

        return total_loss, recon_loss, kl_loss


class BetaScheduler:
    """
    Implements KL Annealing. Gradually increases the beta weight from start_value 
    to end_value over a set number of warmup epochs. This forces the network to 
    learn good reconstructions before enforcing the Gaussian prior on the latent space.
    """
    def __init__(self, start_value=0.0, end_value=1.0, warmup_epochs=20):
        self.start_value = start_value
        self.end_value = end_value
        self.warmup_epochs = warmup_epochs

    def get_beta(self, current_epoch):
        """
        Calculates the beta value for the current epoch.
        """
        if current_epoch >= self.warmup_epochs:
            return self.end_value
            
        # Linear interpolation
        step = (self.end_value - self.start_value) / self.warmup_epochs
        current_beta = self.start_value + (step * current_epoch)
        return current_beta


class TotalVariationLoss(nn.Module):
    """
    (Optional) Total Variation (TV) Regularization.
    In electron tomography, adding a small TV loss to the reconstructed 
    sinogram can reduce high-frequency noise and striping artifacts.
    """
    def __init__(self, weight=1e-4):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        """
        x shape: (B, C, H, W)
        """
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size(1) * t.size(2) * t.size(3)