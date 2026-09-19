import os

CONFIG = {
    "data": {
        "dataset_path": "./cDDPM_v2/dataset/synthetic_raw/",
        "noise_threshold" : 0.3,
        "image_dims": (368, 368),                          # 362x362 -> 368x368: the model can reduce dimensions properly (padding)
        "train_samples": 4000,
        "val_samples": 1000
    },

    "acquisition": {
        "tilt_bounds": (10, 50),                            # sample a max tilt between +/- 10 and 50
        "projection_bounds": (4, 20),                       # sample between 4 and 20 total views
        "views_per_object": 1                               # the model is trained on each object 2 times per epoch (with different configurations)
    },

    "physics": {
        "detector_pixels": 362,                             # actual width of the spatial slice
        "geometry_type": "parallel",                        # standard 2D ET
        "backend": "skimage",           
        "raw_angles" : (-90, 90, 1)                         # sinogram angles and step
    },          

    "model": {          
        "in_channels": 2,                                   # 1 noisy latent (x_t) + 1 conditioning image (FBP)
        "out_channels": 1,                                  # model predicts the single-channel final image
        "base_channels": 32,                                # feature map resolution
        "channel_multipliers": (1, 2, 4, 8),                # multipliers for U-Net downsampling blocks
        "attention_resolutions": (16, 8),                   # spatial resolutions at which cross-attention is applied
    },

    "diffusion": {
        "num_timesteps": 1000,
        "schedule": "cosine",
    },

    "training": {
        "num_workers" : 4,
        "epochs": 70,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,                   # effective batch size = batch_size * grad. accumulation steps
        "use_gradient_clipping" : True,                     # avoid exploding gradients
        "learning_rate": 1e-4,  
        "warmup_epochs": 7, 
        "min_lr": 1e-6,                                     # minimum LR at the end of cosine decay
        "cfg_prob": 0.12,
        "save_frequency": 1,
        "vis_frequency" : 1,
        "output_dir": "./cDDPM_v2/checkpoints/",
        "log_dir": "./cDDPM_v2/logs/",
        "resume_checkpoint": None
    },

    "inference" : {
        "use_projector_guidance" : False,
        "projector_guidance_lambda" : 0.05
    }
}

if __name__ == "__main__":

    from cDDPM_v2.train import train_model

    print("Initializing 2D ET Diffusion Pipeline")
    print(f"Dataset Path: {CONFIG['data']['dataset_path']}")
    print(f"Model Input Channels: {CONFIG['model']['in_channels']}")
    
    os.makedirs(CONFIG["training"]["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["training"]["log_dir"], exist_ok=True)

    print("\nStarting training process...")
    train_model(CONFIG)
