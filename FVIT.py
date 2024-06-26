import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch.nn.functional as F
import wandb
import pytorch_lightning as pl
import librosa
import numpy as np
import collections
import torchaudio

from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from pytorch_lightning.loggers import WandbLogger
from typing import Optional
from timm.models import vision_transformer
from timm.models.vision_transformer import LayerScale, DropPath
from timm.layers import Mlp
from transformers import ViTImageProcessor
# from audio_utils.common.feature_transforms import SpectrogramParser


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x


# class FourierTransformLayer(nn.Module):
#     def forward(self, x):
#         return torch.fft.fftn(x).real


class FViTBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FNetBlock(
        )
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class FViTRunner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = vision_transformer.VisionTransformer(
            img_size=hparams.img_size,
            patch_size=hparams.patch_size,
            in_chans=1,
            num_classes=hparams.n_outputs,
            depth = hparams.depth,
            num_heads = hparams.num_heads,
            drop_rate= hparams.drop_rate,
            qkv_bias=False,
            block_fn=FViTBlock
        )

        # log params

        self.save_hyperparameters()
        self.max_epochs = hparams.max_epochs
        self.learning_rate = hparams.learning_rate
        self.accuracy = Accuracy(
            task='multiclass', num_classes=hparams.n_outputs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # testing metrics
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=3e-2)
        #warmup scheduler
        scheduler = OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps = self.max_epochs*1711)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  
            }
        }


class LogMelSpec(nn.Module):
    def __init__(
        self,
        sr=22050, # 22050, 48000
        n_mels=80, # 80, 120
        n_fft=441, # 441, 1400
        win_len=441, # 441, 1400
        hop_len=220, # 220, 300
        f_min=50., 
        f_max=8000.,
        normalize=True,
        flip_ft=True,
        num_frames=None
    ) -> None:
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize
        self.flip_ft = flip_ft
        if num_frames is not None:
            self.num_frames = int(num_frames)
        else:
            self.num_frames = None

    def forward(self, x):
        if x.ndim == 1:
            # x = x.unsqueeze(0)
            x = np.expand_dims(x, 0)
        # x = self.melspec(x)
        x = librosa.feature.melspectrogram(
            y=x,
            sr=self.sr,
            n_fft=self.n_fft,
            win_length=self.win_len, 
            hop_length=self.hop_len,
            n_mels=self.n_mels, 
            power=2., 
            fmin=self.f_min, 
            fmax=self.f_max
            )
        x = torch.from_numpy(x)
        x = (x + torch.finfo().eps).log()
        if self.num_frames is not None:
            x = x[:, :, :self.num_frames]
        # print("in LogMelSpec, x.shape after melspec", x.shape)
        if self.normalize:
            mean = torch.mean(x, [1, 2], keepdims=True)
            # print("in LogMelSpec, mean.shape", mean.shape)
            std = torch.std(x, [1, 2], keepdims=True)
            x = (x - mean) / (std + 1e-8)
        if self.flip_ft:
            x = x.transpose(-2, -1)
            # print("in LogMelSpec, x.shape after flip_ft", x.shape)
        x = x.unsqueeze(1)
        # print("in LogMelSpec, x.shape after normalization", x.shape)
        return x

class SpeechCommandsDataset(Dataset):
    def __init__(self, directory, label_csv, samplingrate):
        self.directory = directory
        self.filepaths = list(os.path.join(directory, f)
                            for f in os.listdir(directory) if f.endswith('.wav'))
        self.label_to_idx = self.load_label_mapping(label_csv)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=2)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=2)
        self.sampling_rate = samplingrate
    def load_label_mapping(self, label_csv):
        df = pd.read_csv(label_csv)
        label_to_idx = df.set_index('label')['idx'].to_dict()
        # Get the index of the 'unknown' class
        unknown_idx = label_to_idx['_unknown_']
        # Create a default dictionary that returns the 'unknown' index for missing labels
        label_to_idx = collections.defaultdict(
            lambda: unknown_idx, label_to_idx)
        return label_to_idx

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        logmel = LogMelSpec()
        
        data, sr = librosa.load(filepath, sr=self.sampling_rate)
        if sr == 0:
            sr = self.sampling_rate
        
        data = logmel(data)
        data = self.freq_mask(data)
        data = self.time_mask(data)
        data = data.squeeze(1)
        label = self.get_label_from_filepath(filepath)
        label = self.label_to_idx[label]
        return data, label

    @staticmethod
    def get_label_from_filepath(filepath):
        filename = os.path.basename(filepath)
        label = filename.split('_')[0]
        return label

class Lightning_DM(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, label_path, batch_size, n_workers, sampling_rate):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.label_path = label_path
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.sampling_rate = sampling_rate

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_data = SpeechCommandsDataset(self.train_dir, self.label_path, self.sampling_rate)
        self.val_data = SpeechCommandsDataset(self.val_dir, self.label_path, self.sampling_rate)
        self.test_data = SpeechCommandsDataset(self.test_dir, self.label_path, self.sampling_rate)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, num_workers = self.n_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size = self.batch_size, num_workers = self.n_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size, num_workers = self.n_workers)


def main():
    wandb.init()
    # various hyperparameters

    logparams = {
        'img_size': [101,80], # [101,80] [161, 140]
        'patch_size': [20, 2], # [20, 2], [40, 1]
        'exp_name': 'Jakob ViT Run', 
        'sampling_rate': 22050,  # 22050, 48000 for SpeechCommands
        'n_outputs': 12,  # 12, 35 for SpeechCommands
        'wandb_project': 'ViT',
        'gpus': 1,  
        'max_epochs': 50,  
        'learning_rate': 1e-2,
        'batch_size': 50,  
        'n_workers': 9,  
        'depth': 12,
        'num_heads': 12,
        'drop_rate': 0.1,
    }
    hparams = Namespace(**logparams)

    # define logger and model
    wandb_logger = WandbLogger(
        project=hparams.wandb_project,
        log="all",
        name=hparams.exp_name,
        config=logparams
    )

    fvit_lightning = FViTRunner(hparams)
    train_path = '/home/student.aau.dk/jogo18/Speech_Commands/train/'
    test_path =  '/home/student.aau.dk/jogo18/Speech_Commands/test/'
    val_path = '/home/student.aau.dk/jogo18/Speech_Commands/valid/'
    label_path = '/home/student.aau.dk/jogo18/Speech_Commands/labelvocabulary.csv'

    datamodule = Lightning_DM(
        train_dir=train_path,
        test_dir=test_path, 
        val_dir=val_path, 
        label_path=label_path,
        batch_size = hparams.batch_size,
        n_workers = hparams.n_workers,
        sampling_rate = hparams.sampling_rate
        )

# Define Trainer
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="gpu",
        devices="auto",
        logger=wandb_logger,
        # callbacks= [pl.callbacks.EarlyStopping(monitor="val_loss", mode="min")]
    )
    trainer.fit(
        model=fvit_lightning,
        datamodule=datamodule
    )

    trainer.test(
        model=fvit_lightning,
        dataloaders=datamodule
    )
    wandb.finish()
if __name__ == '__main__':
    main()