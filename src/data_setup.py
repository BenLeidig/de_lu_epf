import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl

torch.set_float32_matmul_precision('high')


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
        X_train = torch.tensor(pd.read_pickle(self.X_train_path).to_numpy(), dtype=torch.float32)
        X_val = torch.tensor(pd.read_pickle(self.X_val_path).to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(pd.read_pickle(self.y_train_path).to_numpy()[:, self.imf-1], dtype=torch.float32)
        y_val = torch.tensor(pd.read_pickle(self.y_val_path).to_numpy()[:, self.imf-1], dtype=torch.float32)

        self.train_dataset = EPFDataset(X_train, y_train)
        self.val_dataset = EPFDataset(X_val, y_val)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True
        )