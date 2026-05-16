import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinogramDecoder(nn.Module):
    """
    Decodes the latent vector back into a high-resolution 1-channel sinogram.
    Uses Resize-Convolution to avoid checkerboard artifacts.
    """
    def __init__(self, latent_dim=256, target_size=(181, 512)):
        """
        Args:
            latent_dim (int): Size of the latent bottleneck.
            target_size (tuple): The (Height, Width) of the ground truth complete sinogram.
                                 e.g., (181 angles, 512 detector pixels).
        """
        super().__init__()
        self.target_size = target_size
        
        # ResNet downsamples by a factor of 32 total (2^5).
        # We need to compute the spatial size of the feature map right after the dense layer.
        self.init_h = math.ceil(target_size[0] / 32)
        self.init_w = math.ceil(target_size[1] / 32)
        self.init_channels = 256
        
        # 1. Project latent vector to a flattened spatial tensor
        self.fc = nn.Linear(latent_dim, self.init_channels * self.init_h * self.init_w)
        
        # 2. Upsampling Blocks (Resize + Conv)
        self.upconv1 = self._make_upconv_block(self.init_channels, 128) # 2x size
        self.upconv2 = self._make_upconv_block(128, 64)                 # 4x size
        self.upconv3 = self._make_upconv_block(64, 32)                  # 8x size
        self.upconv4 = self._make_upconv_block(32, 16)                  # 16x size
        self.upconv5 = self._make_upconv_block(16, 16)                  # 32x size
        
        # 3. Final projection to 1 channel (grayscale)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def _make_upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            # Padding=1 with kernel=3 keeps the spatial dimensions intact after upsampling
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, z):
        """
        Args:
            z: Latent space vector of shape (B, latent_dim)
        """
        # 1. Project and Reshape
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_h, self.init_w)
        
        # 2. Upsample 32x
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        
        # At this point, x is slightly larger than target_size because we used math.ceil
        # 3. Interpolate/Crop exactly to the required sinogram dimensions
        if x.shape[2:] != self.target_size:
            # Bilinear interpolation gently snaps the tensor to the exact target size 
            # (e.g., from 192x512 down to 181x512)
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
            
        # 4. Final Conv and Sigmoid
        x = self.final_conv(x)
        x = torch.sigmoid(x) # Output is guaranteed to be [0, 1]
        
        return x