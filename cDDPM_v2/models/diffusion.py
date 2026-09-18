import torch
import torch.nn as nn
import numpy as np
import math

from cDDPM_v2.config import CONFIG


def apply_projector_guidance(x_0_pred, true_sinogram, physics_op, angles, uncertainty_map, lambda_step=0.0):

    """
    Enforces data consistency by taking a weighted gradient step towards the true measured sinogram.
    """

    device = x_0_pred.device
    
    # Prepare network prediction
    x_0_np = torch.clamp(x_0_pred.squeeze(), -1.0, 1.0)
    x_0_np = ((x_0_np + 1.0) / 2.0).cpu().numpy()
    
    # Standard projector gradient step
    sim_sinogram = physics_op.forward_project(x_0_np, angles)
    error_sinogram = true_sinogram - sim_sinogram
    error_img = physics_op.filtered_back_project(error_sinogram, angles)
    
    x_0_projected_np = x_0_np + (lambda_step * error_img)
    x_0_projected_np = np.clip(x_0_projected_np, 0.0, 1.0)
    
    # Convert back to tensors in [0, 1] range to apply the blend
    x_0_pred_tensor = torch.from_numpy(x_0_np).unsqueeze(0).unsqueeze(0).to(device)
    x_0_projected_tensor = torch.from_numpy(x_0_projected_np).unsqueeze(0).unsqueeze(0).to(device)
    
    # Apply Uncertainty Weighting
    # High uncertainty (u -> 1) heavily weights the network prior (x_0_pred)
    # Low uncertainty (u -> 0) heavily weights the data consistency (x_0_projected)
    x_blended = (uncertainty_map * x_0_pred_tensor) + ((1.0 - uncertainty_map) * x_0_projected_tensor)
    
    # Shift back to DDPM [-1, 1] scale
    x_0_updated_tensor = (x_blended * 2.0) - 1.0
    
    return x_0_updated_tensor.to(dtype=torch.float32)


def _extract(a, t, x_shape):
    """
    Extracts values from a 1D tensor 'a' at given indices 't' and reshapes them 
    for broadcasting across the batch tensor 'x'.
    
    Args:
        a (torch.Tensor): The 1D tensor of schedules (e.g., alphas_cumprod).
        t (torch.Tensor): A batch of timestep indices, shape (Batch,).
        x_shape (tuple): The shape of the target tensor (Batch, Channels, Height, Width).
        
    Returns:
        torch.Tensor: The extracted values reshaped to (Batch, 1, 1, 1).
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, config):
        """
        Initializes the DDPM noise schedules and variance buffers.
        
        Args:
            config (dict): The global configuration dictionary from config.py.
        """
        super().__init__()
        
        diff_cfg = config["diffusion"]
        self.num_timesteps = diff_cfg.get("num_timesteps", 1000)
        schedule = diff_cfg.get("schedule", "cosine")
        beta_start = diff_cfg.get("beta_start", 1e-4)
        beta_end = diff_cfg.get("beta_end", 0.02)
        
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        elif schedule == "cosine":
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps, dtype=torch.float32)
            
            # Compute f(t)
            s = 0.008
            f_t = torch.cos(((x / self.num_timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
            
            # Normalize alphas_cumprod to start exactly at 1.0
            alphas_cumprod = f_t / f_t[0]
            
            # Derive betas: beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            
            # Clip betas to prevent singularities near T
            betas = torch.clip(betas, 0.0001, 0.999)

        else:
            raise ValueError(f"Unknown diffusion schedule: {schedule}")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # save all obtained values
        # Register buffers: tensors are automatically moved to the device (CPU/GPU)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Calculations for forward diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for reverse diffusion p(x_{t-1} | x_t)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # We clip to 1e-20 to avoid log(0) at t=0
        self.register_buffer("posterior_variance", torch.clamp(posterior_variance, min=1e-20))

        # Coefficients for x_0 clipping in the reverse process
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt((1.0 / alphas_cumprod) - 1.0))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        """
        The Forward Process: Add noise to the clean image x_0 at timestep t.
        
        Args:
            x_0 (torch.Tensor): The clean ground-truth image, shape (B, 1, H, W).
            t (torch.Tensor): A batch of timesteps, shape (B,).
            noise (torch.Tensor, optional): Pre-sampled Gaussian noise. Defaults to None.
            
        Returns:
            torch.Tensor: The noisy image x_t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alpha_bar_t = _extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # Reparameterization trick: x_t = sqrt(\bar{\alpha}_t) * x_0 + sqrt(1 - \bar{\alpha}_t) * \epsilon
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t

    @torch.no_grad()
    def p_sample(self, model, x_t, x_fbp, acq_config, t, t_index, uncertainty_map, true_sinogram=None, physics_op=None, angles=None, guidance_scale=2.0):
        """
        The Reverse Process (Single Step) using intermediate x_0 clipping.
        """
        # Conditional noise prediction (with x_fbp and acq_config)
        noise_cond = model(x_t, x_fbp, t, acq_config)
        
        # Unconditional noise prediction (null condition)
        x_fbp_null = torch.zeros_like(x_fbp)
        acq_config_null = torch.zeros_like(acq_config)
        noise_uncond = model(x_t, x_fbp_null, t, acq_config_null)
        
        # Apply Classifier-Free Guidance extrapolation
        # Matches the formula: (1-s)*uncond + s*cond
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # Predict the clean image (x_0) from x_t and the extrapolated noise
        sqrt_recip_alphas_cumprod_t = _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        x_0_pred = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise_pred

        # --- NEW: Data Consistency Projector ---
        if true_sinogram is not None and physics_op is not None and angles is not None:
            x_0_pred = apply_projector_guidance(
                x_0_pred, 
                true_sinogram, 
                physics_op, 
                angles,
                uncertainty_map,
                lambda_step=CONFIG["inference"]["projector_guidance_lambda"]
            )
        
        # Compute the posterior mean using the CLAMPED x_0
        posterior_mean_coef1_t = _extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = _extract(self.posterior_mean_coef2, t, x_t.shape)
        
        model_mean = posterior_mean_coef1_t * x_0_pred + posterior_mean_coef2_t * x_t
        
        # If we are at the very last step (t=0), return the mean
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = _extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    @torch.no_grad()
    def p_sample_loop(self, model, x_fbp, acq_config, uncertainty_map, true_sinogram=None, physics_op=None, angles=None, guidance_scale=2.0):
        """
        The Complete Reverse Process: Generates a sample from pure noise given x_fbp.
        (Primarily used for inference/validation).
        
        Args:
            model (nn.Module): The Measurement-Conditioned U-Net.
            x_fbp (torch.Tensor): The deterministic FBP initialization.
            
        Returns:
            torch.Tensor: The final reconstructed image x_0.
        """
        device = x_fbp.device
        b, c, h, w = x_fbp.shape
        
        # Start from pure Gaussian noise
        x_t = torch.randn((b, c, h, w), device=device)
        
        # Iterate backwards from T-1 down to 0
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, x_fbp, acq_config, t, i, uncertainty_map, true_sinogram, physics_op, angles, guidance_scale)
            
        return x_t