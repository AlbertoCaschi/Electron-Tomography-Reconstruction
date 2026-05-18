import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinogramDecoder(nn.Module):
    """
    Decodes the latent vector back into a high-resolution 1-channel sinogram.
    Uses Resize-Convolution to avoid checkerboard artifacts.
    """
    def __init__(self, latent_dim=1024, target_size=(181, 362)):
        """
        Args:
            latent_dim (int): Size of the latent bottleneck.
            target_size (tuple): The (Height, Width) of the ground truth complete sinogram (181, 362).
        """
        super().__init__()
        self.target_size = target_size
        
        # ResNet downsamples by a factor of 32 total (2^5)
        # compute the spatial size of the feature map after the fully connected layer
        self.init_h = math.ceil(target_size[0] / 32)
        self.init_w = math.ceil(target_size[1] / 32)
        self.init_channels = 256
        
        # projects latent vector to a flattened spatial tensor
        self.fc = nn.Linear(latent_dim, self.init_channels * self.init_h * self.init_w)
        
        # upsampling Blocks (Resize + Conv)
        self.upconv1 = self._make_upconv_block(self.init_channels, 128) # 2x size
        self.upconv2 = self._make_upconv_block(128, 64)                 # 4x size
        self.upconv3 = self._make_upconv_block(64, 32)                  # 8x size
        self.upconv4 = self._make_upconv_block(32, 16)                  # 16x size
        self.upconv5 = self._make_upconv_block(16, 16)                  # 32x size
        
        # final projection to 1 channel
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
        # project and reshape (prepare array to be fed into up_convolution layers)
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_h, self.init_w)
        
        # upsampling
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        
        # x is slightly larger than target_size because we used math.ceil

        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='nearest')
        
        '''
        # we perform a center crop to obtain an image of the same dimensions as the input
        if x.shape[2:] != self.target_size:
            _, _, h, w = x.shape
            target_h, target_w = self.target_size
            
            top_offset = (h - target_h) // 2
            left_offset = (w - target_w) // 2
            
            x = x[:, :, top_offset : top_offset + target_h, left_offset : left_offset + target_w]
        '''
        
        x = self.final_conv(x)
        # with sigmoid the output is guaranteed to be [0, 1]
        x = torch.sigmoid(x)
        
        return x