import os

# Centralized configuration dictionary for the 2D Electron-Tomography Pipeline
CONFIG = {
    # --- Data Parameters ---
    "data": {
        "dataset_path": "./cDDPM/dataset/synthetic_raw/",  # Directory containing your .mrc files
        "image_dims": (368, 368),                          # Target dimensions (height, width)
        "train_samples": 2500,                             # Number of files for the training set
        "val_samples": 500                                 # Number of files for the validation set
    },

    # --- Acquisition Configurations ---
    "acquisition": {
        "tilt_bounds": (40, 60),          # Will sample a max tilt between +/- 40 and 60
        "projection_bounds": (8, 20),     # Will sample between 8 and 20 total views
        "views_per_object": 5            # Train/Validate on each object 10 times per epoch
    },

    # --- Physics Parameters ---
    "physics": {
        "detector_pixels": 362,                    # Typically matches the width of the spatial slice
        "geometry_type": "parallel",               # Assuming parallel-beam geometry for standard 2D ET
        "backend": "skimage",                      # Designates the engine (e.g., ASTRA toolbox)
        "raw_angles" : (-90, 90, 1)                # sinograms angles and step
    },

    # --- Model Parameters (cDDPM) ---
    "model": {
        "in_channels": 2,                          # Early fusion: 1 noisy latent (x_t) + 1 conditioning image (FBP)
        "out_channels": 1,                         # Network predicts the single-channel added noise
        "base_channels": 32,                       # Starting feature map resolution
        "channel_multipliers": (1, 2, 4, 8),       # Multipliers for U-Net downsampling blocks
        "attention_resolutions": (16, 8),          # Spatial resolutions at which to apply cross-attention
    },

    # --- Diffusion Parameters ---
    "diffusion": {
        "num_timesteps": 1000,                     # Total T steps for the forward/reverse process
        "schedule": "cosine",                      # Variance schedule
    },

    # --- Training Parameters ---
    "training": {
        "epochs": 30,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,                # 4 x 2 = Effective batch size of 8
        "learning_rate": 1e-4,
        "warmup_epochs": 5,                              # Number of epochs to warm up the LR
        "min_lr": 1e-6,                                  # Minimum LR at the end of cosine decay
        "save_frequency": 1,                             # Save model checkpoint every N epochs
        "vis_frequency" : 1,
        "output_dir": "./cDDPM/checkpoints/",            # Directory for saved model weights
        "log_dir": "./cDDPM/logs/",                      # Directory for TensorBoard/logging outputs
        "resume_checkpoint": r"C:\Users\Alberto\Desktop\Electron-Tomography-Reconstruction\cDDPM\checkpoints\unet_checkpoint_interrupted.pt"
    }
}

if __name__ == "__main__":

    # Importing locally to avoid circular dependencies if train.py also imports CONFIG
    try:
        from cDDPM.train import train_model
    except ImportError:
        print("Warning: 'train.py' not found or 'train_model' not implemented yet.")
        train_model = None

    print("=== Initializing 2D ET Diffusion Pipeline ===")
    print(f"Dataset Path: {CONFIG['data']['dataset_path']}")
    print(f"Model Input Channels: {CONFIG['model']['in_channels']} (Early Fusion Enabled)")
    
    # Ensure necessary output directories exist before training starts
    os.makedirs(CONFIG["training"]["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["training"]["log_dir"], exist_ok=True)

    if train_model:
        print("\nStarting training process...")
        train_model(CONFIG)
    else:
        print("\nSetup complete. Ready to implement the training loop.")