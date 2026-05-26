import numpy as np
import os
import mrcfile
from pathlib import Path
from train import train_model

# Main config dictionary
CONFIG = {
    "experiment_name": "vae_resnet18_baseline",
    "checkpoint_dir": "./VAEResNet/checkpoints/",
    "log_dir": "./VAEResNet/logs/",

    # detector width of 362 pixels
    "full_angle_min": -90,
    "full_angle_max": 90,
    "target_size": (181, 362),
    
    # experimental setups
    "acquisition_configs": [
        {'range': (-90, 90), 'step': 1},
        {'range': (-50, 50), 'step': 5},
        {'range': (-50, 50), 'step': 10},
        {'range': (-50, 50), 'step': 20},
        {'range': (-40, 40), 'step': 5},
        {'range': (-40, 40), 'step': 10},
        {'range': (-40, 40), 'step': 20}
    ],

    # Model Architecture
    "resnet_type": "resnet18", # resnet34
    "freeze_early_layers": False,
    "latent_dim": 64,

    # Training Hyperparameters
    "batch_size": 16,
    "num_workers": 4,
    "learning_rate": 5e-5, # AdamW optimizer
    "num_epochs": 200,
    "early_stopping_patience": 50,  # stops after 50 epochs without val_loss improvement
    "resume_training" : True,      # True -> resumes previous training
    
    # Loss and VAE Settings
    "recon_loss_type": "l1",        # 'l1' (MAE) is critical for sharp sinograms
    "kl_warmup_epochs": 75,         # Gradually increase KL Divergence over 75 epochs
    
    # Visualization
    "viz_freq": 5                   # generates visualizations every 5 epochs
}



if __name__ == "__main__":
    print(f"--- Initializing Experiment: {CONFIG['experiment_name']} ---")
    
    # update paths
    CONFIG['checkpoint_dir'] = os.path.join(CONFIG['checkpoint_dir'], CONFIG['experiment_name'])
    CONFIG['log_dir'] = os.path.join(CONFIG['log_dir'], CONFIG['experiment_name'])

    print("Loading Ground Truth Sinograms from MRC files...")

    data_folder = Path("./VAEResNet/dataset/synthetic_raw")
    mrc_file_paths = list(data_folder.iterdir())

    loaded_sinos = []
    
    for path in mrc_file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find MRC file at: {path}")

        with mrcfile.open(path, permissive=True) as mrc:
            # extracts as numpy array
            data = mrc.data.copy().astype(np.float32)
            
            if data.ndim == 3 and data.shape[0] == 1:
                data = np.squeeze(data, axis=0)
                
            # data must be saved as (Angles, Detector)
            if data.shape[0] == CONFIG['target_size'][1]:
                 data = data.T

            assert data.shape == CONFIG['target_size'], \
                f"Shape mismatch in {path}! Expected {CONFIG['target_size']}, got {data.shape}"

            loaded_sinos.append(data)

    # Training and validation splitting
    train_sinos = loaded_sinos[:2500]
    val_sinos = loaded_sinos[2500:3000]
   

    print(f"Loaded {len(train_sinos)} training objects and {len(val_sinos)} validation object.")
    

    train_model(CONFIG, train_sinos, val_sinos)