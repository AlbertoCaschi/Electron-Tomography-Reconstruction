import math
import pytorch_lightning as pl
import torch
import tqdm
import yaml
from torch import nn

from .fourier import apply_fourier_mask_to_patch
from .masked_loss import masked_loss
from .missing_wedge import get_missing_wedge_mask
from .normalization import get_avg_model_input_mean_and_std_from_dataloader


class LitUnet2D(pl.LightningModule):
    """
    PyTorchLightning wrapper.
    """

    def __init__(
        self,
        unet_params,
        adam_params,
        patch_dir,
        update_patch_missing_wedges_every_n_epochs=10,
    ):
        super().__init__()
        self.unet_params = unet_params
        self.adam_params = adam_params
        self.patch_dir = patch_dir
        self.update_patch_missing_wedges_every_n_epochs = update_patch_missing_wedges_every_n_epochs
        
        # Instantiate the new ResNet Encoder-Decoder instead of the U-Net
        self.unet = ResNetEncoderDecoder2D(**self.unet_params)
        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x.unsqueeze(1)).squeeze(1)

    def training_step(self, batch, batch_idx):
        model_output = self(batch["model_input"])
        
        loss = masked_loss(
            model_output=model_output,
            target=batch["model_target"],
            rot_mw_mask=batch["rot_mw_mask"],
            mw_mask=batch["mw_mask"],
            mw_weight=100.0  # Adjusted missing wedge weight
        )
        self.log("fitting_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self(batch["model_input"])
        loss = masked_loss(
            model_output=model_output,
            target=batch["model_target"],
            rot_mw_mask=batch["rot_mw_mask"],
            mw_mask=batch["mw_mask"],
            mw_weight=100.0
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_train_start(self) -> None:
        if self.current_epoch == 0:
            self.update_normalization()

    def on_train_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.update_patch_missing_wedges_every_n_epochs == 0:
            self.update_patch_missing_wedges()
            self.update_normalization()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.adam_params)
        return [optimizer]

    def update_patch_missing_wedges(self):
        datasets = []
        train_loader = self.trainer.train_dataloader.loaders
        train_set = train_loader.dataset
        train_set.rotate_patches = False 
        datasets.append(train_set)
        
        if self.trainer.val_dataloaders is not None:
            val_loader = self.trainer.val_dataloaders[0]
            val_set = val_loader.dataset
            val_set.rotate_patches = False
            datasets.append(val_set)
            
        dataset = torch.utils.data.ConcatDataset(datasets)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=train_loader.batch_size, num_workers=train_loader.num_workers,
        )
        
        patch_dim = dataset[0]["model_input"].shape[-1]
        factor = 2 ** self.unet_params["num_downsample_layers"]
        padding = factor * math.ceil(patch_dim / factor) - patch_dim
        
        mw_mask = get_missing_wedge_mask(grid_size=2*[patch_dim + padding], mw_angle=train_set.mw_angle)
        
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc="Updating patch missing wedges"):
                patch_batch = batch["model_input"].to(self.device)
                patch_batch = torch.nn.functional.pad(patch_batch, pad=(0, padding, 0, padding), mode="constant", value=0)
                mw_mask_batch = mw_mask.repeat((*patch_batch.shape[:-2], 1, 1)).to(patch_batch.device)
                
                patch_batch_ref = self.forward(patch_batch)
                
                patch_batch = apply_fourier_mask_to_patch(patch_batch, mw_mask_batch) + \
                              apply_fourier_mask_to_patch(patch_batch_ref, 1 - mw_mask_batch)
                              
                patch_batch = patch_batch[..., :patch_dim, :patch_dim]
                
                for patch, file in zip(patch_batch, batch["patch0_file"]):
                    torch.save(patch.cpu().clone(), file)
                    
        train_set.rotate_patches = True
        if self.trainer.val_dataloaders is not None:
            val_set.rotate_patches = True

    def update_normalization(self):
        loc, scale = get_avg_model_input_mean_and_std_from_dataloader(self.trainer.train_dataloader, verbose=True)
        self.unet.normalization_loc = loc
        self.unet.normalization_scale = scale
        self.unet_params["normalization_loc"] = loc
        self.unet_params["normalization_scale"] = scale
        self.update_hparam("unet_params", self.unet_params)

    def update_hparam(self, hparam, value):
        logger = self.trainer.logger
        logdir = f"{logger.save_dir}/{logger.name}/version_{logger.version}"
        hparams_file = f"{logdir}/hparams.yaml"
        hparams = yaml.safe_load(open(hparams_file, "r"))
        hparams[hparam] = value
        with open(hparams_file, "w") as f:
            yaml.dump(hparams, f)



# RESNET ENCODER-DECODER ARCHITECTURE


class ResNetEncoderDecoder2D(torch.nn.Module):

    def __init__(
        self, in_chans=1, out_chans=1, chans=64, num_downsample_layers=3, drop_prob=0.0, 
        residual=True, normalization_loc=0.0, normalization_scale=1.0
    ):
        super().__init__()
        self.residual = residual
        
        self.normalization_loc = normalization_loc
        self.normalization_scale = normalization_scale

        # Initial Projection
        self.initial_conv = nn.Conv2d(in_chans, chans, kernel_size=3, padding=1)

        # Encoder (Downsampling + ResBlocks)
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = chans
        for _ in range(num_downsample_layers):
            self.encoders.append(ResBlock2D(ch, ch, drop_prob))
            self.downsamplers.append(SpatialDownSampling(ch))
            ch_next = ch * 2
            self.encoders.append(ResBlock2D(ch, ch_next, drop_prob))
            ch = ch_next

        # Bottleneck (Deepest latent space)
        self.bottleneck = nn.Sequential(
            ResBlock2D(ch, ch, drop_prob),
            ResBlock2D(ch, ch, drop_prob)
        )

        # Decoder (Upsampling)
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(num_downsample_layers):
            ch_next = ch // 2
            self.upsamplers.append(SpatialUpSampling(ch, ch_next))
            self.decoders.append(ResBlock2D(ch_next, ch_next, drop_prob))
            ch = ch_next

        # Final Output Projection
        self.final_conv = nn.Conv2d(ch, out_chans, kernel_size=1)

    @property
    def normalization_loc(self):
        return self._normalization_loc

    @normalization_loc.setter
    def normalization_loc(self, loc):
        self._normalization_loc = nn.Parameter(
            torch.tensor(loc, dtype=torch.float32), requires_grad=False
        )

    @property
    def normalization_scale(self):
        return self._normalization_scale

    @normalization_scale.setter
    def normalization_scale(self, scale):
        self._normalization_scale = nn.Parameter(
            torch.tensor(scale, dtype=torch.float32), requires_grad=False
        )

    def normalize(self, volume):
        return (volume - self.normalization_loc) / (self.normalization_scale + 1e-6)

    def denormalize(self, volume):
        return volume * (self.normalization_scale + 1e-6) + self.normalization_loc

    def forward(self, volume):
        x = self.normalize(volume)
        out = self.initial_conv(x)

        # Encode
        for enc1, down, enc2 in zip(self.encoders[0::2], self.downsamplers, self.encoders[1::2]):
            out = enc1(out)
            out = down(out)
            out = enc2(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decode
        for up, dec in zip(self.upsamplers, self.decoders):
            out = up(out)
            out = dec(out)

        out = self.final_conv(out)

        if self.residual:
            out = out + volume

        return self.denormalize(out)


class ResBlock2D(nn.Module):
    """ Standard Residual Block with InstanceNorm """
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.match_dim = nn.Conv2d(in_chans, out_chans, kernel_size=1) if in_chans != out_chans else nn.Identity()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_chans),
            nn.Dropout2d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_chans)
        )
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)

    def forward(self, x):
        res = self.match_dim(x)
        return self.activation(self.layers(x) + res)


class SpatialDownSampling(nn.Module):
    def __init__(self, chans: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(chans, chans, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume):
        return self.layers(volume)


class SpatialUpSampling(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        # No more 'cat' logic! Pure upsampling.
        self.tconv = nn.ConvTranspose2d(
            in_chans, out_chans, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.activation(self.tconv(volume))