import torch
import torch.nn as nn
from torchvision import models

class ResNetVAEEncoder(nn.Module):
    """
    ResNet-based Encoder for the Tomography Variational Autoencoder.
    Maps a 1-channel sinogram to the mean (mu) and log-variance (logvar) of the latent space.
    """
    def __init__(self, latent_dim=1024, resnet_type='resnet18', pretrained=True, freeze_early_layers=True):
        super().__init__()
        self.latent_dim = latent_dim
        
        # load ResNet
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

        # the first convolutional layer is changed to handle sinograms of 1 channel
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
            # to preserve pre-trained weights we sum the RGB weights along the channel dimension
            # output = (w_r + w_g + w_b) * x
            # instead of w_r * x + w_g * x + w_b * x (we would need to change the data to have 3 channels)
            with torch.no_grad():
                self.base_model.conv1.weight[:] = torch.sum(old_conv1.weight, dim=1, keepdim=True)

        # keep all the convolution layers and remove the last two (avgpool and fc)
        modules = list(self.base_model.children())[:-2] 
        self.feature_extractor = nn.Sequential(*modules)

        # freeze early layers
        if freeze_early_layers:
            self._freeze_early_layers()

        # we will use AvgPool2d to force the spatial dimensions to 1x1 before the
        # FC layer (latent space)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # mu and logvar of the probability distribution of the autoencoder
        self.fc_mu = nn.Linear(fc_in_features, latent_dim)
        self.fc_logvar = nn.Linear(fc_in_features, latent_dim)

    def _freeze_early_layers(self):
        """
        Freezes the early layers (conv1, bn1, layer1, layer2) to act as a 
        generic feature extractor, leaving layer3 and layer4 trainable to 
        learn the physics/geometry of the sinograms.
        """
        # freeze everything
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # unfreeze the deeper layers
        for layer_name in ['7', '8']: # 7 is layer3, 8 is layer4
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