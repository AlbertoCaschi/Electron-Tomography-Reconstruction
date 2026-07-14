import torch
import torch.nn as nn
from VAEResNet.models.encoder import ResNetVAEEncoder
from VAEResNet.models.decoder import SinogramDecoder

class TomographyVAE(nn.Module):
    """
    The full Variational Autoencoder pipeline for Sinogram Inpainting.
    Combines the pre-trained ResNet encoder and the custom upsampling decoder.
    """
    def __init__(self, latent_dim=256, target_size=(181, 512), 
                 resnet_type='resnet18', freeze_early_layers=True):
        super().__init__()
        
        self.encoder = ResNetVAEEncoder(
            latent_dim=latent_dim, 
            resnet_type=resnet_type, 
            pretrained=True, 
            freeze_early_layers=freeze_early_layers
        )
        
        self.decoder = SinogramDecoder(
            latent_dim=latent_dim, 
            target_size=target_size
        )

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick: z = mu + std * epsilon
        Allows gradients to flow back through the stochastic sampling process.
        """
        if self.training:
            # sigma = exp(0.5 * log_variance)
            std = torch.exp(0.5 * logvar)
            # epsilon sampling from standard normal distribution
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # during inference/validation we use the mean to remove noise
            return mu

    def forward(self, x):
        """
        Args:
            x: Masked/Noisy Input Sinogram (B, 1, H, W)
            
        Returns:
            recon_x: Reconstructed Complete Sinogram (B, 1, target_H, target_W)
            mu: Latent mean
            logvar: Latent log variance
        """

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar