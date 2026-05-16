import torch
import torch.nn as nn
from torchvision import models

class ResNetVAEEncoder(nn.Module):
    """
    ResNet-based Encoder for the Tomography Variational Autoencoder.
    Maps a 1-channel sinogram to the mean (mu) and log-variance (logvar) of the latent space.
    """
    def __init__(self, latent_dim=256, resnet_type='resnet18', pretrained=True, freeze_early_layers=True):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 1. Load the base ResNet model
        if resnet_type == 'resnet18':
            # Use the new weights syntax if pretrained is True
            weights = 'DEFAULT' if pretrained else None
            self.base_model = models.resnet18(weights=weights)
            conv1_out_channels = 64
            fc_in_features = 512

        elif resnet_type == 'resnet34':
            self.base_model = models.resnet34(pretrained=pretrained)
            conv1_out_channels = 64
            fc_in_features = 512
        else:
            raise ValueError(f"Unsupported resnet_type: {resnet_type}. Choose 'resnet18' or 'resnet34'.")

        # 2. Handle the 1-channel input (Grayscale Sinogram)
        old_conv1 = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=old_conv1.out_channels, 
            kernel_size=old_conv1.kernel_size, 
            stride=old_conv1.stride, 
            padding=old_conv1.padding, 
            bias=old_conv1.bias is not None
        )
        
        if pretrained:
            # Trick to preserve pre-trained weights: Sum the RGB weights along the channel dimension
            with torch.no_grad():
                self.base_model.conv1.weight[:] = torch.sum(old_conv1.weight, dim=1, keepdim=True)

        # 3. Strip the final Fully Connected layer and Average Pool
        # We keep everything up to the final convolutional feature map
        modules = list(self.base_model.children())[:-2] 
        self.feature_extractor = nn.Sequential(*modules)

        # 4. Freeze early layers if requested
        if freeze_early_layers:
            self._freeze_early_layers()

        # 5. Latent Space Projections
        # ResNet18/34 feature maps are [B, 512, H/32, W/32]
        # We will use AdaptiveAvgPool2d to force the spatial dimensions to 1x1 before the FC layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_mu = nn.Linear(fc_in_features, latent_dim)
        self.fc_logvar = nn.Linear(fc_in_features, latent_dim)

    def _freeze_early_layers(self):
        """
        Freezes the early layers (conv1, bn1, layer1, layer2) to act as a 
        generic feature extractor, leaving layer3 and layer4 trainable to 
        learn the physics/geometry of the sinograms.
        """
        # First, freeze everything
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Then, unfreeze the deeper layers
        for layer_name in ['7', '8']: # In the Sequential module, 7 is layer3, 8 is layer4
            if int(layer_name) < len(self.feature_extractor):
                for param in self.feature_extractor[int(layer_name)].parameters():
                    param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: Masked/Noisy Sinogram Tensor of shape (B, 1, H, W)
        Returns:
            mu: Latent space mean of shape (B, latent_dim)
            logvar: Latent space log variance of shape (B, latent_dim)
        """
        # Extract features: Output shape -> (B, 512, H_out, W_out)
        features = self.feature_extractor(x)
        
        # Pool spatial dimensions to 1x1: Output shape -> (B, 512, 1, 1)
        pooled_features = self.adaptive_pool(features)
        
        # Flatten for Dense layers: Output shape -> (B, 512)
        flattened = torch.flatten(pooled_features, 1)
        
        # Project to latent space bottleneck
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)
        
        return mu, logvar