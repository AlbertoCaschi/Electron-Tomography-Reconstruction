import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class ConditionalUNet(nn.Module):
    def __init__(self, config):
        """
        Initializes the Measurement-Conditioned U-Net for early fusion.
        """
        super().__init__()
        
        model_cfg = config["model"]
        in_channels = model_cfg.get("in_channels", 3)
        out_channels = model_cfg.get("out_channels", 1)
        base_channels = model_cfg.get("base_channels", 64) 
        channel_multipliers = model_cfg.get("channel_multipliers", (1, 2, 4, 8))
        
        self.max_tilt = float(config["acquisition"]["tilt_bounds"][1])
        self.max_proj = float(config["acquisition"]["projection_bounds"][1])
        
        block_out_channels = tuple([base_channels * m for m in channel_multipliers])

        # In diffusers, the internal time embedding dimension is always base_channels * 4
        time_embed_dim = base_channels * 4

        # The MLP now outputs the exact time_embed_dim to smoothly fuse with the timestep
        self.acq_embedder = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, time_embed_dim)
        )

        # All cross-attention blocks are removed in favor of standard spatial blocks
        down_block_types = (
            "DownBlock2D",              
            "DownBlock2D",              
            "DownBlock2D",              
            "DownBlock2D",     
        )
        
        up_block_types = (
            "UpBlock2D",       
            "UpBlock2D",                
            "UpBlock2D",                
            "UpBlock2D",                
        )
        
        self.unet = UNet2DConditionModel(
            sample_size=config["data"]["image_dims"],
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            mid_block_type="UNetMidBlock2D",
            class_embed_type="identity",
        )

    def forward(self, x_t, x_sirt, x_unc, timestep, acq_config):
        """
        The forward pass executing the early fusion conditioning.
        """
        if x_t.shape[1] != 1 or x_sirt.shape[1] != 1 or x_unc.shape[1] != 1:
            raise ValueError(
                f"Expected 1-channel inputs for x_t, x_sirt, and x_unc, "
                f"got {x_t.shape[1]}, {x_sirt.shape[1]}, and {x_unc.shape[1]} channels instead."
            )
            
        fused_input = torch.cat([x_t, x_sirt, x_unc], dim=1)

        acq_config_normalized = acq_config.clone()
        acq_config_normalized[:, 0] = acq_config[:, 0] / self.max_tilt
        acq_config_normalized[:, 1] = acq_config[:, 1] / self.max_proj

        cond_embeds = self.acq_embedder(acq_config_normalized)
        
        # Pass the dummy tensor as the required positional argument
        x_0_pred = self.unet(
                fused_input,
                timestep,
                encoder_hidden_states=None, 
                class_labels=cond_embeds
            ).sample
        
        return x_0_pred