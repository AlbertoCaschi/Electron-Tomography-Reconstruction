import torch
import torch.nn as nn
import math


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
    def p_sample(self, model, x_t, x_fbp, t, t_index):
        """
        The Reverse Process (Single Step) using intermediate x_0 clipping.
        """
        # Predict the noise using our early fusion U-Net
        noise_pred = model(x_t, x_fbp, t)
        
        # Predict the clean image (x_0) from x_t and the predicted noise
        sqrt_recip_alphas_cumprod_t = _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        x_0_pred = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise_pred
        
        # Clamp the predicted x_0 to strictly remain in grayscale bounds
        x_0_pred = torch.clamp(x_0_pred, min=-1.0, max=1.0)
        
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
    def p_sample_loop(self, model, x_fbp):
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
            x_t = self.p_sample(model, x_t, x_fbp, t, i)
            
        return x_t