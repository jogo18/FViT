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

from timm.models import VisionTransformer

# from FNET.FNET_pytorch import FNetBlock


# Define Dataset

Speech_Commands_Dataset = ds.load_dataset('speech_commands', 'v0.02')
classes = 34


# various parameters

logparams = {
    'exp_name': 'Jakob Fourier Project Test',  # unique name for model and logs
    # sampling rate
    'sampling_rate': Speech_Commands_Dataset['train'][0]['audio']['sampling_rate'],
    'n_outputs': classes,
    'wandb_project': 'FourierViT',
    'gpus': 1,  # number of gpus
    'max_epochs': 1,  # number of times during training, where the whole dataset is traversed
    'learning_rate': 1e-3,
    'batch_size': 1,  # should be considered together with learning rate. decrease if using a small machine and getting memory errors
    'n_workers': 2,  # set to 0 in windows when working with a windows on a small machine
}

hparams = Namespace(**logparams)


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x


class FourierTransformLayer(nn.Module):
    def forward(self, x):
        return torch.fft.fftn(x).real


# train_data = DataLoader(
#     Speech_Commands_Dataset['train'],
#     hparams.batch_size,
#     shuffle=True,
#     num_workers=1
# )

# path = '/home/jagrole/AAU/8.Semester/Project/Code/Data/backward_3291330e_nohash_2.wav'

# waveform, sample_rate = torchaudio.load(path, normalize=True)

# testdata = train_data[3000]['audio']['array']
# testdata = torch.tensor(testdata)
# transform = torchaudio.transforms.Spectrogram(n_fft=800)

# print(train_data[3000])

# D = librosa.stft(testdata)
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# plt.figure().set_figwidth(12)
# librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
# plt.colorbar()
# plt.show()
# class LitModel(pl.LightningModule):
#     def __init__(self, hparams):
#         super().__init__()
#         self.model = VisionTransformer.VisionTransformer()

#         # log params

#         self.save_hyperparameters()
#         self.learning_rate = hparams.learning_rate
#         self.accuracy = Accuracy(
#             task='multiclass', num_classes=hparams.n_outputs)

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self.forward(x)
#         weights = torch.tensor([0.2, 0.8], device=self.device)
#         loss = torch.nn.functional.cross_entropy(logits, y, weights)

#         # training metrics
#         preds = torch.argmax(logits, dim=1)
#         acc = self.accuracy(preds, y)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
#         self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         weights = torch.tensor([0.2, 0.8]).to(self.device)
#         loss = torch.nn.functional.cross_entropy(logits, y, weights)

#         # validation metrics
#         preds = torch.argmax(logits, dim=1)
#         acc = self.accuracy(preds, y)
#         self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
#         self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)

#         # testing metrics
#         loss = torch.nn.functional.cross_entropy(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         acc = self.accuracy(preds, y)
#         self.log('test_loss', loss, prog_bar=True)
#         self.log('test_acc', acc, prog_bar=True)

#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         return optimizer


# def main():
#     #set random seed
#     pl.seed_everything(1)

#     #define transforms for augmentation
#     if hparams.transformtype == 1:
#         train_transforms = transforms.Compose([
#             transforms.ToTensor()
#         ]
#         )

#     # define dataset
#     dataset = Dataset(
#         audio_list=data['path'],
#         labels=labels,
#         transforms=train_transforms
#     )

#     train_dataset = 0
#     val_dataset = 0
#     test_dataset = 0

#     # Dataloaders for training and validation
#     train_load = DataLoader(train_dataset, hparams.batch_size, shuffle = True, num_workers = 4)
#     val_load = DataLoader(val_dataset, hparams.batch_size, shuffle = False, num_workers = 4)
#     test_load = DataLoader(test_dataset, hparams.batch_size, shuffle = False, num_workers = 4)


#     # define logger and model
#     wandb_logger = WandbLogger(
#         project=hparams.wandb_project,
#         log_model="all",
#         name = hparams.exp_name,
#         config = logparams
#         )

#     lightningtest = LitModel(hparams)

# #Define Trainer
#     trainer = pl.Trainer(
#         limit_train_batches= len(train_dataset)//hparams.batch_size,
#         max_epochs=hparams.max_epochs,
#         accelerator="gpu",
#         devices="auto",
#         logger=wandb_logger,
#     )


#     wandb.finish()

# if __name__ == '__main__':
#     main()
