import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class ConditionalUNet(nn.Module):
    def __init__(self, config):
        """
        Initializes the Measurement-Conditioned U-Net for early fusion.
        
        Args:
            config (dict): The global configuration dictionary from config.py.
        """
        super().__init__()
        
        model_cfg = config["model"]
        in_channels = model_cfg.get("in_channels", 2)
        out_channels = model_cfg.get("out_channels", 1)
        base_channels = model_cfg.get("base_channels", 64)
        channel_multipliers = model_cfg.get("channel_multipliers", (1, 2, 4, 8))
        
        # Calculate the exact channel dimensions for each block based on the multipliers
        block_out_channels = tuple([base_channels * m for m in channel_multipliers])

        # Define cross-attention dimension and a small MLP embedder for the config
        cross_attention_dim = 256
        self.acq_embedder = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, cross_attention_dim)
        )

        # For a 4-level U-Net, we inject spatial attention at the lower resolutions (e.g., 3rd level)
        # to help the model learn global structural coherence.
        down_block_types = (
            "DownBlock2D",              # Level 1: 368x368 -> 184x184
            "DownBlock2D",              # Level 2: 184x184 -> 92x92
            "DownBlock2D",              # Level 3: 92x92   -> 46x46
            "CrossAttnDownBlock2D",     # Level 4: Cross attention applied
        )
        
        up_block_types = (
            "CrossAttnUpBlock2D",       # Level 4: Upsampling with CrossAttention at 46x46
            "UpBlock2D",                # Level 3: Upsampling
            "UpBlock2D",                # Level 2: Upsampling
            "UpBlock2D",                # Level 1: Upsampling
        )
        
        # Instantiate the core diffusion U-Net from the Hugging Face library
        self.unet = UNet2DConditionModel(
            sample_size=config["data"]["image_dims"],
            in_channels=in_channels,
            out_channels=out_channels,
            cross_attention_dim=cross_attention_dim,
            layers_per_block=2,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )

    def forward(self, x_t, x_fbp, timestep, acq_config):
        """
        The forward pass executing the early fusion conditioning.
        
        Args:
            x_t (torch.Tensor): The noisy image at step t, shape [Batch, 1, H, W]
            x_fbp (torch.Tensor): The static FBP initialization, shape [Batch, 1, H, W]
            timestep (torch.Tensor): The current diffusion timesteps, shape [Batch]
            
        Returns:
            torch.Tensor: The predicted noise, shape [Batch, 1, H, W]
        """
        # Ensure the inputs have the expected channel dimensions
        if x_t.shape[1] != 1 or x_fbp.shape[1] != 1:
            raise ValueError(
                f"Expected 1-channel inputs for x_t and x_fbp, "
                f"got {x_t.shape[1]} and {x_fbp.shape[1]} channels instead."
            )
            
        # Concatenate the noisy state and the deterministic FBP condition along the channel dimension (dim=1). 
        # Resulting shape: [Batch, 2, Height, Width]
        fused_input = torch.cat([x_t, x_fbp], dim=1)

        # Map [Batch, 2] to sequence format [Batch, Sequence_Length, cross_attention_dim]
        cond_embeds = self.acq_embedder(acq_config).unsqueeze(1)
        
        # Pass the fused input and the time embeddings into the U-Net
        # diffusers returns an output object; the predicted tensor is stored in .sample
        noise_pred = self.unet(
                fused_input,
                timestep,
                encoder_hidden_states=cond_embeds
            ).sample
        
        return noise_pred