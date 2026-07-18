import torch

'''
checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': current_val_loss,
                'best_val_loss': best_val_loss
            }
'''

checkpoint_path = "./cDDPM/checkpoints/unet_checkpoint_interrupted.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# manually overwrite the checkpoint parameter that needs to be changed
checkpoint['epoch'] = 4 

torch.save(checkpoint, checkpoint_path)
print("Successfully updated the current checkpoint")