import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

# Import our custom modules
from models.vae import TomographyVAE
from data.dataset import TomographyDataset
from data.transforms import PairedCompose, RandomHorizontalShift, RandomHorizontalFlip, RandomIntensityScale, ToTensor
from utils.losses import VAELoss, BetaScheduler
from utils.reconstruction import batch_reconstruct
from utils.viz import plot_reconstruction_dashboard, plot_training_curves

def train_model(config, train_sinos, val_sinos):
    """
    Main training loop for the Tomography VAE.
    
    Args:
        config (dict): Configuration dictionary (hyperparameters, paths, etc.)
        train_sinos (list of np.ndarray): Ground truth sinograms for training (Objects 1, 2, 3)
        val_sinos (list of np.ndarray): Ground truth sinograms for validation (Object 4)
    """
    # 1. Setup Device & Directories
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"--- Starting Training on {device} ---")
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    train_transform = PairedCompose([])
    """ # 2. Data Transforms
    train_transform = PairedCompose([
        RandomHorizontalShift(max_shift=20, p=0.7),
        RandomHorizontalFlip(p=0.5),
        RandomIntensityScale(scale_range=(0.8, 1.2), p=0.5),
    ]) """
    
    # Validation strictly has no spatial augmentations
    val_transform = PairedCompose([])

    # 3. Datasets and DataLoaders
    base_angles_deg = np.linspace(config['full_angle_min'], config['full_angle_max'], config['target_size'][0])
    
    train_dataset = TomographyDataset(train_sinos, config['acquisition_configs'], 
                                      base_angles_deg=base_angles_deg, is_training=True, transform=train_transform)
    val_dataset = TomographyDataset(val_sinos, config['acquisition_configs'], 
                                    base_angles_deg=base_angles_deg, is_training=False, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # 4. Initialize Model, Loss, Scheduler, and Optimizer
    model = TomographyVAE(
        latent_dim=config['latent_dim'], 
        target_size=config['target_size'], 
        resnet_type=config['resnet_type'],
        freeze_early_layers=config['freeze_early_layers']
    ).to(device)

    criterion = VAELoss(recon_loss_type=config['recon_loss_type'])
    beta_scheduler = BetaScheduler(start_value=0.0, end_value=0.0, warmup_epochs=config['kl_warmup_epochs'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 5. Tracking Variables
    train_losses, val_losses, beta_history = [], [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    try:
        # 6. Main Epoch Loop
        for epoch in range(1, config['num_epochs'] + 1):
            # --- TRAINING ---
            model.train()
            train_loss_epoch = 0.0
            current_beta = beta_scheduler.get_beta(epoch)
            
            # Use tqdm for a nice progress bar
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']} [Train]")
            for inputs, targets in train_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                recon_x, mu, logvar = model(inputs)
                
                loss, recon_loss, kl_loss = criterion(recon_x, targets, mu, logvar, beta=current_beta)
                loss.backward()
                
                # Gradient clipping prevents exploding gradients in VAEs
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss_epoch += loss.item()
                train_pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Recon': f"{recon_loss.item():.4f}"})
                
            avg_train_loss = train_loss_epoch / len(train_loader)
            train_losses.append(avg_train_loss)
            beta_history.append(current_beta)

            # --- VALIDATION ---
            model.eval()
            val_loss_epoch = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{config['num_epochs']} [Val]")
                for inputs, targets in val_pbar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    recon_x, mu, logvar = model(inputs)
                    loss, recon_loss, kl_loss = criterion(recon_x, targets, mu, logvar, beta=current_beta)
                    
                    val_loss_epoch += loss.item()
                    val_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                    
            avg_val_loss = val_loss_epoch / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Step the Learning Rate Scheduler
            lr_scheduler.step(avg_val_loss)

            print(f"Epoch {epoch} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Beta: {current_beta:.3f}")

            # --- VISUALIZATION (Every N epochs) ---
            if epoch % config['viz_freq'] == 0:
                print("Generating Visualizations...")
                sample_in = inputs[0]
                sample_tgt = targets[0]
                sample_pred = recon_x[0]

                # Reconstruct to 2D using FBP
                recon_in_2d = batch_reconstruct(sample_in.unsqueeze(0), base_angles_deg, method='fbp')[0]
                recon_tgt_2d = batch_reconstruct(sample_tgt.unsqueeze(0), base_angles_deg, method='fbp')[0]
                recon_pred_2d = batch_reconstruct(sample_pred.unsqueeze(0), base_angles_deg, method='fbp')[0]

                # Plot and save
                plot_path = os.path.join(config['log_dir'], f"dashboard_epoch_{epoch}.png")
                plot_reconstruction_dashboard(sample_in, sample_tgt, sample_pred,
                                              recon_in_2d, recon_tgt_2d, recon_pred_2d,
                                              epoch=epoch, save_path=plot_path)

            # --- CHECKPOINTING & EARLY STOPPING ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                
                # Save the best model
                ckpt_path = os.path.join(config['checkpoint_dir'], "best_vae_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss
                }, ckpt_path)
                print(f"-> Saved new best model to {ckpt_path}")
            else:
                epochs_no_improve += 1
                print(f"-> No improvement for {epochs_no_improve} epochs.")
                
            if epochs_no_improve >= config['early_stopping_patience']:
                print("!!! Early Stopping Triggered !!!")
                break

    except KeyboardInterrupt:
        print("\n\n----- Training interrupted by user (Ctrl+C). Saving progress so far... -----")

    # plot training curves
    if train_losses:
        print("Generating loss curves...")
        loss_curve_path = os.path.join(config['log_dir'], "training_curves.png")
        plot_training_curves(train_losses, val_losses, beta_values=beta_history, save_path=loss_curve_path)
        
        # save losses to CSV file
        csv_path = os.path.join(config['log_dir'], "training_losses.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Beta'])
            for i in range(len(train_losses)):
                writer.writerow([i + 1, train_losses[i], val_losses[i], beta_history[i]])
        print(f"Saved losses to {csv_path}")
    else:
        print("Training was interrupted before completing the first epoch. No curves or CSV generated.")

    print("Done!")

# ==========================================
# Placeholder for testing the script directly
# ==========================================
if __name__ == "__main__":
    print("This script is meant to be run via config.py or a main entry point.")
    print("Waiting for config setup to proceed...")