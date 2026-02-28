from joblib import dump
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.vmd import VmdTransformer
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl

torch.set_float32_matmul_precision('high')


def save_splits(df:pd.DataFrame, time_splits:list, f_names:list, data_dir:str='../data/processed/', scaler_dir:str='../models/scalers/'):
    for (train_start, train_end, test_start, test_end), (f_train, f_test) in zip(time_splits, f_names):
        
        df_train = df.loc[train_start:train_end]
        df_test = df.loc[test_start:test_end]

        ss = StandardScaler()
        df_train_scaled = pd.DataFrame(ss.fit_transform(df_train), columns=df_train.columns)
        df_test_scaled = pd.DataFrame(ss.transform(df_test), columns=df_test.columns)

        dump(ss, scaler_dir+'ss_'+f_train+'.pkl')
        df_train.to_pickle(data_dir+f_train+'.pkl')
        df_test.to_pickle(data_dir+f_test+'.pkl')
        df_train_scaled.to_pickle(data_dir+f_train+'_scaled.pkl')
        df_test_scaled.to_pickle(data_dir+f_test+'_scaled.pkl')


def save_imf_splits(df:pd.DataFrame, time_splits:list, f_names:list, data_dir:str='../data/processed/', scaler_dir:str='../models/scalers/', vmd_dir:str='../models/vmd/'):
    for (train_start, train_end, test_start, test_end), (f_train, f_test) in zip(time_splits, f_names):
        
        # instantiation
        vmd = VmdTransformer(K=5, alpha=4_000)
        vmd.set_random_state(0)
        ss = StandardScaler()

        # train-test split
        df_train = df.loc[train_start:train_end]
        np_train_price = df_train['price'].to_numpy()
        df_test = df.loc[test_start:test_end]
        np_test_price = df_test['price'].to_numpy()

        # VMD train
        np_train_imf = vmd.fit_transform(np_train_price)
        df_train_imf = pd.DataFrame(data=np_train_imf, columns=[f'imf{i}' for i in range(1, 6)])
        df_train_imf['resid'] = np_train_price - df_train_imf.sum(axis=1)
        df_train_imf = df_train_imf.set_index(df_train.index)
        # scale train
        np_train_imf_scaled = ss.fit_transform(df_train_imf)
        df_train_imf_scaled = pd.DataFrame(data=np_train_imf_scaled, columns=df_train_imf.columns)
        df_train_imf_scaled = df_train_imf_scaled.set_index(df_train.index)

        # VMD test
        np_test_imf = vmd.transform(np_test_price)
        df_test_imf = pd.DataFrame(data=np_test_imf, columns=[f'imf{i}' for i in range(1, 6)])
        df_test_imf['resid'] = np_test_price - df_test_imf.sum(axis=1)
        df_test_imf = df_test_imf.set_index(df_test.index)
        # scale test
        np_test_imf_scaled = ss.transform(df_test_imf)
        df_test_imf_scaled = pd.DataFrame(data=np_test_imf_scaled, columns=df_test_imf.columns)
        df_test_imf_scaled = df_test_imf_scaled.set_index(df_test.index)

        # exporting
        dump(vmd, vmd_dir+'vmd_'+f_train+'.pkl')
        dump(ss, scaler_dir+'ss_'+f_train+'.pkl')
        df_train_imf.to_pickle(data_dir+f_train+'.pkl')
        df_test_imf.to_pickle(data_dir+f_test+'.pkl')
        df_train_imf_scaled.to_pickle(data_dir+f_train+'_scaled.pkl')
        df_test_imf_scaled.to_pickle(data_dir+f_test+'_scaled.pkl')


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
        X_train = pd.read_pickle(self.X_train_path).to_numpy()
        X_val = pd.read_pickle(self.X_val_path).to_numpy()
        y_train = pd.read_pickle(self.y_train_path).to_numpy()
        y_val = pd.read_pickle(self.y_val_path).to_numpy()
        
        X_train = np.concatenate([X_train, y_train], axis=1)
        X_val = np.concatenate([X_val, y_val], axis=1)

        self.train_dataset = EPFDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train[:, self.imf-1], dtype=torch.float32)
        )
        self.val_dataset = EPFDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val[:, self.imf-1], dtype=torch.float32)
        )
    
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