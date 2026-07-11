import os

# Centralized configuration dictionary for the 2D Electron-Tomography Pipeline
CONFIG = {
    # --- Data Parameters ---
    "data": {
        "dataset_path": "./dataset/synthetic_raw/",  # Directory containing your .mrc files
        "image_dims": (368, 368),                  # Target dimensions (height, width)
    },

    # --- Acquisition Configurations ---
    # Reflects the different angular ranges and step sizes available in the dataset
    "acquisition_configs": [
        {'range': (-90, 90), 'step': 1},
        {'range': (-50, 50), 'step': 5},
        {'range': (-50, 50), 'step': 10},
        {'range': (-50, 50), 'step': 20},
        {'range': (-40, 40), 'step': 5},
        {'range': (-40, 40), 'step': 10},
        {'range': (-40, 40), 'step': 20}
    ],

    # --- Physics Parameters ---
    "physics": {
        "detector_pixels": 362,                    # Typically matches the width of the spatial slice
        "geometry_type": "parallel",               # Assuming parallel-beam geometry for standard 2D ET
        "backend": "astra",                        # Designates the engine (e.g., ASTRA toolbox)
    },

    # --- Model Parameters (cDDPM) ---
    "model": {
        "in_channels": 2,                          # Early fusion: 1 noisy latent (x_t) + 1 conditioning image (FBP)
        "out_channels": 1,                         # Network predicts the single-channel added noise
        "base_channels": 64,                       # Starting feature map resolution
        "channel_multipliers": (1, 2, 4, 8),       # Multipliers for U-Net downsampling blocks
        "attention_resolutions": (16, 8),          # Spatial resolutions at which to apply cross-attention
    },

    # --- Diffusion Parameters ---
    "diffusion": {
        "num_timesteps": 1000,                     # Total T steps for the forward/reverse process
        "schedule": "linear",                      # Variance schedule (options: 'linear', 'cosine')
        "beta_start": 1e-4,
        "beta_end": 0.02,
    },

    # --- Training Parameters ---
    "training": {
        "epochs": 100,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "save_frequency": 10,                      # Save model checkpoint every N epochs
        "output_dir": "./checkpoints/",            # Directory for saved model weights
        "log_dir": "./logs/",                      # Directory for TensorBoard/logging outputs
    }
}

if __name__ == "__main__":

    from train import train_model
    # Importing locally to avoid circular dependencies if train.py also imports CONFIG
    try:
        from train import train_model
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