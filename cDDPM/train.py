import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.optim as optim
from tqdm import tqdm

from cDDPM.data.dataset import TomographyDataset
from cDDPM.models.unet import ConditionalUNet
from cDDPM.models.diffusion import GaussianDiffusion
from cDDPM.utils.visualization import plot_training_curves, save_reconstruction_progress



def train_model(config):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        

    print(f"Using device: {device}")
    

    print("Loading datasets...")
    train_dataset = TomographyDataset(config, mode="train")
    val_dataset = TomographyDataset(config, mode="val")
    
    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], 
        shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], 
        shuffle=False, num_workers=0, drop_last=False
    )
    
    
    # the first element of the val set is used to visualize progress every epoch
    print("Extracting fixed validation sample for progress tracking...")
    fixed_x_0, fixed_x_fbp = val_dataset[0]
    fixed_x_0 = fixed_x_0.unsqueeze(0)
    fixed_x_fbp = fixed_x_fbp.unsqueeze(0)

    # model
    unet = ConditionalUNet(config).to(device)
    diffusion = GaussianDiffusion(config).to(device)
    
    optimizer = optim.AdamW(unet.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-4)
    criterion = nn.MSELoss()

    epochs = config["training"]["epochs"]
    warmup_epochs = config["training"].get("warmup_epochs", 5)
    min_lr = config["training"].get("min_lr", 1e-6)
    
    # LR scheduler
    # from 10% of base_lr to 100% of base_lr over the warmup epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    # cosine decay from base_lr down to min_lr over the remaining epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(epochs - warmup_epochs), eta_min=min_lr)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler], # chain schedulers
        milestones=[warmup_epochs] # switch
        )


    save_freq = config["training"]["save_frequency"]
    vis_freq = config["training"].get("vis_frequency", 1)
    output_dir = config["training"]["output_dir"]
    log_dir = config["training"]["log_dir"]
    num_timesteps = config["diffusion"]["num_timesteps"]
    resume_path = config["training"].get("resume_checkpoint", None)
    
    # Resume Logic
    start_epoch = 1
    best_val_loss = float('inf')
    
    if resume_path and os.path.exists(resume_path):
        print(f"\nLoading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        elif 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
            
        print(f"Resuming at Epoch {start_epoch}. Current Best Val Loss: {best_val_loss:.6f}")
    

    # CSV Setup
    csv_log_path = os.path.join(log_dir, "training_log.csv")
    if start_epoch == 1 or not os.path.exists(csv_log_path):
        with open(csv_log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Val Loss"])
            
    print("\n=== Starting Training ===")
    print("Press Ctrl+C at any time to safely save the model and stop.\n")
    
    current_epoch = start_epoch - 1
    current_val_loss = float('inf') 

    try:
        # epoch Loop
        for epoch in range(start_epoch, epochs + 1):
            current_epoch = epoch
            
            accum_steps = config["training"].get("gradient_accumulation_steps", 1)
            
            ## TRAIN ##

            unet.train()
            train_loss = 0.0
            
            train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
            
            optimizer.zero_grad()
            
            for batch_idx, (x_0, x_fbp) in train_progress: # x_0 -> ground truth
                x_0 = x_0.to(device, dtype=torch.float32)
                x_fbp = x_fbp.to(device, dtype=torch.float32)
                
                t = torch.randint(0, num_timesteps, (x_0.shape[0],), device=device).long()
                noise = torch.randn_like(x_0)
                x_t = diffusion.q_sample(x_0, t, noise=noise)
                
                noise_pred = unet(x_t, x_fbp, t)
                
                loss = criterion(noise_pred, noise)
                loss = loss / accum_steps
                
                loss.backward()
                
                # step optimizer when a complete batch is done (depending on accumulation steps)
                if ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):

                    if config["training"]["use_gradient_clipping"]:
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)

                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += (loss.item() * accum_steps)
                train_progress.set_postfix({"loss": f"{(loss.item() * accum_steps):.4f}"})
                
            avg_train_loss = train_loss / len(train_loader)
            

            ## VALIDATION ##

            unet.eval()
            val_loss = 0.0
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False)
            
            with torch.no_grad():
                for x_0, x_fbp in val_progress:
                    x_0 = x_0.to(device, dtype=torch.float32)
                    x_fbp = x_fbp.to(device, dtype=torch.float32)
                    
                    t = torch.randint(0, num_timesteps, (x_0.shape[0],), device=device).long()
                    noise = torch.randn_like(x_0)
                    x_t = diffusion.q_sample(x_0, t, noise=noise)
                    
                    noise_pred = unet(x_t, x_fbp, t)
                    loss = criterion(noise_pred, noise)
                    val_loss += loss.item()
                    
            current_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {current_val_loss:.6f}")
            
            # CSV logging
            with open(csv_log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, current_val_loss])
            
            # create checkpoint
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': current_val_loss,
                'best_val_loss': best_val_loss
            }
            
            if epoch % save_freq == 0 or epoch == epochs:
                torch.save(checkpoint_data, os.path.join(output_dir, f"unet_checkpoint_epoch_{epoch}.pt"))
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(checkpoint_data, os.path.join(output_dir, "unet_checkpoint_best.pt"))
                print(f"--> New best model saved! (Val Loss: {best_val_loss:.6f})")

            # generate visualization
            if epoch % vis_freq == 0 or epoch == epochs:
                save_reconstruction_progress(
                    unet, diffusion, fixed_x_0, fixed_x_fbp, 
                    epoch, log_dir, device
                )


            current_lr = scheduler.get_last_lr()[0]
            print(f"--> Current Learning Rate: {current_lr:.6f}")
            
            scheduler.step()


    except KeyboardInterrupt:
        print("\n\n[Interrupt] Training stopped by user (Ctrl+C).")
        interrupted_path = os.path.join(output_dir, "unet_checkpoint_interrupted.pt")
        completed_epoch = current_epoch-1
        torch.save({
            'epoch': completed_epoch,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': current_val_loss,
            'best_val_loss': best_val_loss
        }, interrupted_path)
        print(f"--> Interrupted state saved to: {interrupted_path}")
        print("To resume, update 'resume_checkpoint' in config.py with this path.")
        
    finally:
        print("\nTraining Complete / Halted")
        print("Generating visualization...")
        plot_training_curves(csv_log_path)