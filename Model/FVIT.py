import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch.nn.functional as F
import wandb
import pytorch_lightning as pl
import datasets as ds
import librosa
import numpy as np

from torchaudio import transforms
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from pytorch_lightning.loggers import WandbLogger
from typing import Optional
from timm.models import vision_transformer
from timm.models.vision_transformer import LayerScale, DropPath
from timm.layers import Mlp
from transformers import ViTImageProcessor


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x


class FourierTransformLayer(nn.Module):
    def forward(self, x):
        return torch.fft.fftn(x).real


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
        # self.attn = Attention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_norm=qk_norm,
        #     attn_drop=attn_drop,
        #     proj_drop=proj_drop,
        #     norm_layer=norm_layer,
        # )
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


class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.dataset = None
        self.transform = nn.Sequential(
            transforms.Spectrogram()
        )
        self.Speech_Commands_train = None
        self.Speech_Commands_val = None
        self.Speech_Commands_test = None
        self.batch_size = batch_size

    # def prepare_data(self):
    #     # download data
    #     self.dataset = ds.load_dataset(
    #         'speech_commands', 'v0.02')
    #     self.classes = 34

    #     processor = ViTImageProcessor()
    #     audio = batch["audio"]
    #     batch["input_values"] = processor(
    #         audio["array"], sampling_rate=audio["sampling_rate"]).input_values[-1]
    #     batch["input_length"] = len(batch["input_values"])
    #     with processor.as_target_processor():
    #         batch["labels"] = processor(batch["sentence"]).input_ids
    #     return batch

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.Speech_Commands_train = self.dataset['train']
        self.Speech_Commands_val = self.dataset['validation']
        self.Speech_Commands_test = self.dataset['test']

    def train_dataloader(self):
        return DataLoader(self.Speech_Commands_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.Speech_Commands_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.Speech_Commands_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.Speech_Commands_predict, batch_size=32)


class LitModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = vision_transformer.VisionTransformer(
            img_size=hparams.img_size,
            patch_size=hparams.patch_size,
            in_channels=1,
            num_classes=hparams.n_outputs,
            qkv_bias=False,
            block_fn=FViTBlock
        )

        # log params

        self.save_hyperparameters()
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

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)

    #     # testing metrics
    #     loss = torch.nn.functional.cross_entropy(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     acc = self.accuracy(preds, y)
    #     self.log('test_loss', loss, prog_bar=True)
    #     self.log('test_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def main():

    # various hyperparameters
    logparams = {
        'img_size': [257, 63],
        'patch_size': [16, 7],
        'exp_name': 'Jakob Fourier Project Test',  # unique name for model and logs
        # sampling rate
        'sampling_rate': 16000,  # 16000 for SpeechCommands
        'n_outputs': 34,  # 34 for SpeechCommands
        'wandb_project': 'FourierViT',
        'gpus': 1,  # number of gpus
        'max_epochs': 1,  # number of times during training, where the whole dataset is traversed
        'learning_rate': 1e-3,
        'batch_size': 1,  # should be considered together with learning rate. decrease if using a small machine and getting memory errors
        'n_workers': 4,  # set to 0 in windows when working with a windows on a small machine
    }
    hparams = Namespace(**logparams)
    # set random seed
    # pl.seed_everything(1)
    dm = SpeechCommandsDataModule(batch_size=1)
    temp2 = dm.prepare_data()
    temp = dm.setup(stage='fit')
    test = dm.train_dataloader()
    test2 = next(iter(test))
    print(test2['input_values'])
    # print(test2)
    # modeltest = vision_transformer.VisionTransformer(
    #     img_size=hparams.img_size,
    #     patch_size=hparams.patch_size,
    #     in_chans=1,
    #     num_classes=hparams.n_outputs,
    #     qkv_bias=False,
    #     block_fn=FViTBlock
    # )
    # asd = modeltest(test2['audio']['array'])
    # print(asd)
    # define logger and model
    # wandb_logger = WandbLogger(
    #     project=hparams.wandb_project,
    #     log_model="all",
    #     name=hparams.exp_name,
    #     config=logparams
    # )

    # lightmodule = LitModel(hparams)

# Define Trainer
    # trainer = pl.Trainer(
    #     max_epochs=hparams.max_epochs,
    #     accelerator="gpu",
    #     devices="auto",
    #     logger=wandb_logger,
    # )

    # trainer.fit(
    #     model=lightmodule,
    #     datamodule=dm
    # )

    # trainer.test(
    #     model=lightningtest,
    #     dataloaders=test_load
    # )

    # wandb.finish()


if __name__ == '__main__':
    main()
