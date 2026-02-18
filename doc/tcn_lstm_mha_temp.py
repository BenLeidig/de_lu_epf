import os
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from pytorch_tcn import TCN
import optuna


class EPFDataset(torch.utils.data.Dataset):
    def __init__(self, X:torch.tensor, y:torch.tensor, seq_len:int=24*7*4, pred_len:int=24, stride:int=24):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
    
    def __len__(self):
        return (len(self.X)-self.seq_len-self.pred_len)//self.stride + 1
    
    def __getitem__(self, i):
        num_strides = i*self.stride
        return (
            self.X[num_strides : num_strides+self.seq_len],
            self.y[num_strides+self.seq_len : num_strides+self.seq_len+self.pred_len]
        )
    

class EPFDataModule(pl.LightningDataModule):
    def __init__(
            self,
            X_train_path:str,
            X_val_path:str,
            y_train_path:str,
            y_val_path:str,
            imf:int,
            batch_size:int
        ):
        super().__init__()
        self.X_train_path = X_train_path
        self.X_val_path = X_val_path
        self.y_train_path = y_train_path
        self.y_val_path = y_val_path
        self.imf = imf
        self.batch_size = batch_size

    def setup(self, stage=None):
        with open(self.X_train_path, 'rb') as f:
            X_train = torch.tensor(joblib.load(f), dtype=torch.float32)
        with open(self.X_val_path, 'rb') as f:
            X_val = torch.tensor(joblib.load(f), dtype=torch.float32)
        with open(self.y_train_path, 'rb') as f:
            y_train = torch.tensor(joblib.load(f)[:, self.imf-1], dtype=torch.float32)
        with open(self.y_val_path, 'rb') as f:
            y_val = torch.tensor(joblib.load(f)[:, self.imf-1], dtype=torch.float32)

        self.train_dataset = EPFDataset(X_train, y_train)
        self.val_dataset = EPFDataset(X_val, y_val)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()//2,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()//2,
            persistent_workers=True
        )
    

class TCN_LSTM_MHA(pl.LightningModule):
    def __init__(
        self,
        input_size:int,
        channel_sizes:list,
        kernel_size:int,
        hidden_sizes:list,
        tcn_dropout:float,
        lstm_dropouts:list,
        mha_dropout:float,
        mha_heads:int,
        lr_init:float
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = torch.nn.MSELoss()
        self.lr_init = lr_init

        #### model ####
        # ReLU
        self.relu = nn.ReLU()
        
        # TCN
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=channel_sizes,
            kernel_size=kernel_size,
            dropout=tcn_dropout,
            use_skip_connections=True,
            input_shape='NLC'   # (batch_size, time_steps, feature_channels)
        )

        # LSTM
        self.lstm0 = nn.LSTM(input_size=channel_sizes[-1], hidden_size=hidden_sizes[0], batch_first=True)
        self.dropout0 = nn.Dropout(lstm_dropouts[0])
        self.lstm1 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)
        self.dropout1 = nn.Dropout(lstm_dropouts[1])
        self.lstm2 = nn.LSTM(input_size=hidden_sizes[1], hidden_size=hidden_sizes[2], batch_first=True)
        
        # MHA
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_sizes[-1],
            num_heads=mha_heads,
            dropout=mha_dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_sizes[-1])

        # head
        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        # TCN
        x = self.tcn(x)
        x = self.relu(x)
        
        # LSTM
        x, _ = self.lstm0(x)
        x = self.relu(x)
        x = self.dropout0(x)

        x, _ = self.lstm1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)

        # MHA
        mha_out, _ = self.mha(x, x, x)
        x = self.norm(x+mha_out)

        # head
        return self.fc(x[:, -24:, :]).squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_init)
        return optimizer
    

def objective(trial:optuna.trial.Trial):

    #### general params ####
    seq_len = 24*7*4
    batch_size = trial.suggest_int('batch_size', 16, 64, log=True)
    lr_init = trial.suggest_float('lr_init', 1e-4, 1e-2, log=True)

    #### TCN params ####
    dilation_base = 2
    kernel_size = trial.suggest_int('kernel_size', 2, 8)
    num_blocks = int(
        np.ceil(np.log2( ((seq_len-1)*(dilation_base-1)) / ((kernel_size-1)*2) + 1 ))
    )

    channel_sizes = list(np.zeros(num_blocks, dtype=int))
    for i in range(num_blocks):
        channel_sizes[i] = trial.suggest_int(
            f'channel_size_{num_blocks}_{i}',
            channel_sizes[i-1] if i>0 else 8,
            256
        )

    tcn_dropout = trial.suggest_float('tcn_dropout', 0.0, 0.5)

    #### LSTM params ####
    hidden_size0 = trial.suggest_int(
        'hidden_size0',
        max(16, channel_sizes[-1]//4),
        min(256, channel_sizes[-1])
    )
    hidden_size1 = trial.suggest_int('hidden_size1', 16, min(256, channel_sizes[-1]))
    hidden_size2 = trial.suggest_int('hidden_size2', 16, min(256, channel_sizes[-1]))
    hidden_sizes = [hidden_size0, hidden_size1, hidden_size2]

    lstm_dropout0 = trial.suggest_float('lstm_dropout0', 0.0, 0.5)
    lstm_dropout1 = trial.suggest_float('lstm_dropout1', 0.0, 0.5)
    lstm_dropouts = [lstm_dropout0, lstm_dropout1]

    #### MHA params ####
    mha_dropout = trial.suggest_float('mha_dropout', 0.0, 0.5)
    heads_range = [h for h in [1, 2, 4, 8, 16] if hidden_sizes[-1]%h==0]
    mha_heads = trial.suggest_categorical('mha_heads', heads_range)

    #### callbacks ####
    callbacks = [
        optuna.integration.PyTorchLightningPruningCallback(trial, monitor='val_loss'),
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ]

    #### training ####
    val_losses = np.zeros(3)

    for i in range(1, 4):
        datamodule = EPFDataModule(
            X_train_path=f'../data/processed/train{i}_scaled.pkl',
            X_val_path=f'../data/processed/val{i}_scaled.pkl',
            y_train_path=f'../data/processed/imf_train{i}_scaled.pkl',
            y_val_path=f'../data/processed/imf_val{i}_scaled.pkl',
            imf=1,
            batch_size=batch_size
        )
        mod = TCN_LSTM_MHA(
            input_size=8,
            channel_sizes=channel_sizes,
            kernel_size=kernel_size,
            hidden_sizes=hidden_sizes,
            tcn_dropout=tcn_dropout,
            lstm_dropouts=lstm_dropouts,
            mha_dropout=mha_dropout,
            mha_heads=mha_heads,
            lr_init=lr_init
        )
        
        trainer = pl.Trainer(
            max_epochs=50,
            callbacks=callbacks,
            accelerator='auto',
            logger=False
        )
        trainer.fit(mod, datamodule=datamodule)
        val_losses[i-1] = trainer.callback_metrics['val_loss'].item()

    return np.mean(val_losses)