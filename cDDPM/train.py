import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# Import the custom modules we built
from .data.dataset import TomographyDataset
from .models.unet import ConditionalUNet
from .models.diffusion import GaussianDiffusion

def train_model(config):
    """
    Executes the training loop for the Measurement-Conditioned Diffusion Model.
    
    Args:
        config (dict): The global configuration dictionary from config.py.
    """
    # 1. Device Configuration
    # Automatically use CUDA if available, otherwise fallback to CPU (or MPS for Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"--- Initialization ---")
    print(f"Using device: {device}")
    
    # 2. Data Loading
    print("Loading dataset...")
    train_dataset = TomographyDataset(config, mode="train")
    
    # CRITICAL: num_workers=0 is used to prevent CUDA/multiprocessing context 
    # crashes if the physics operator relies on specific C/C++ backends under the hood.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )
    print(f"Dataset loaded. Total batches per epoch: {len(train_loader)}")
    
    # 3. Model Initialization
    print("Initializing models...")
    unet = ConditionalUNet(config).to(device)
    diffusion = GaussianDiffusion(config).to(device)
    
    # 4. Optimization Setup
    optimizer = optim.AdamW(
        unet.parameters(), 
        lr=config["training"]["learning_rate"], 
        weight_decay=1e-4
    )
    criterion = nn.MSELoss()
    
    # Extract training hyperparameters
    epochs = config["training"]["epochs"]
    save_freq = config["training"]["save_frequency"]
    output_dir = config["training"]["output_dir"]
    num_timesteps = config["diffusion"]["num_timesteps"]
    
    print("\n=== Starting Training ===")
    
    # 5. The Epoch Loop
    for epoch in range(1, epochs + 1):
        unet.train()
        epoch_loss = 0.0
        
        # Use tqdm for a clean progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        # 6. The Batch Loop
        for batch_idx, (x_0, x_fbp) in enumerate(progress_bar):
            # Move data to the active device
            x_0 = x_0.to(device, dtype=torch.float32)
            x_fbp = x_fbp.to(device, dtype=torch.float32)
            
            batch_size = x_0.shape[0]
            
            # Step A: Sample random timesteps uniformly for each image in the batch
            t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
            
            # Step B: Sample true Gaussian noise
            noise = torch.randn_like(x_0)
            
            # Step C: Forward Diffusion (Add noise to the clean ground-truth)
            x_t = diffusion.q_sample(x_0, t, noise=noise)
            
            # Step D: Model Prediction (Early fusion conditioning happens inside unet)
            noise_pred = unet(x_t, x_fbp, t)
            
            # Step E: Loss Calculation & Backpropagation
            loss = criterion(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} Completed | Average Loss: {avg_epoch_loss:.6f}")
        
        # 7. Checkpointing
        if epoch % save_freq == 0 or epoch == epochs:
            checkpoint_path = os.path.join(output_dir, f"unet_checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"--> Saved checkpoint: {checkpoint_path}")

    print("=== Training Complete ===")