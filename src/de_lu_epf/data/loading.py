from pathlib import Path
from typing import Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ArrayLike = Union[np.ndarray, torch.Tensor]


class ANNDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset class for ANN models in this repository. Intended for seq-to-seq forecasting using sliding windows.

    Args:
        X (Union[np.ndarray, torch.Tensor]): Feature matrix.
        y (Union[np.ndarray, torch.Tensor]): Target array.
        seq_len (int): Lookback length.
        pred_len (int): Prediction length.
        stride (int): Amount to skip for each prediction. (Should equal pred_len for this research.)
    """

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        seq_len: int,  # NOTE: recommended 24 * 7 * 4 (4 week lookback)
        pred_len: int,  # NOTE: MUST be 24 (next-day hourly predictions)
        stride: int,  # NOTE: MUST equal pred_len (preds 12:00 / day)
    ):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

    def __len__(self):
        # Essentially, this is the number of samples in the dataset.
        # Logic: we need self.seq_len amount of data on frontend and
        ## self.pred_len amount of data on backend; divide by self.stride
        ## because we 'skip' by that amount each step; add 1 because python
        ## indexes from 0.
        # NOTE: rounded division should impact our scenario since the data
        ## has specific start / end dates perfectly aligned with the start
        ## / end of days.
        # NOTE: the above specification also means we mustn't specifically
        ## check that our provided data is starting / ending on 00:00 and
        ## 23:00, respectively.
        return (len(self.X) - self.seq_len - self.pred_len) // self.stride + 1

    def __getitem__(self, i):
        num_strides = i * self.stride  # Total length to 'skip' until
        return (
            self.X[num_strides : num_strides + self.seq_len],
            self.y[
                num_strides + self.seq_len : num_strides + self.seq_len + self.pred_len
            ],  # Since we are forecasting, our response 'y' is the *next* pred_len steps
        )


class ANNDataModule(pl.LightningDataModule):
    """Custom PytorchLightning DataModule classfor ANN models in this repository.

    Args:
        data_dir (Path): Path to the directory containing data (pandas DataFrames).
        batch_size (int): Length of each batch.
        target_col (str): Name of the target column from each DataFrame.
        seq_len (int): Lookback length.
        pred_len (int): Prediction length.
        stride (int): Amount to skip for each prediction. (Should equal pred_len for this research.)
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        target_col: str,
        seq_len: int = 24 * 7 * 2,
        pred_len: int = 24,
        stride: int = 24,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

    def setup(self, stage=None):
        # NOTE: MUST ALWAYS exist train_scaled.parquet in data_dir path.
        ## Logic: this is to read the index of the the target_col to
        ## avoid unnecessary recomputations.
        df_train = pd.read_parquet(self.data_dir / "train_scaled.parquet")
        self.target_idx = df_train.columns.get_loc(self.target_col)
        self.input_size = df_train.shape[1]

        # Must call ANNDataModule.setup() before calling dataloaders.
        if stage in (
            None,
            "fit",
        ):  ## fit stage constructs train and validation datasets.
            ## Training dataset:
            np_train = df_train.to_numpy()
            self.X_train = np_train
            self.y_train = np_train[:, self.target_idx]
            self.train_dataset = ANNDataset(
                self.X_train, self.y_train, self.seq_len, self.pred_len, self.stride
            )

            ## Validation dataset:
            np_val = pd.read_parquet(self.data_dir / "val_scaled.parquet").to_numpy()
            self.X_val = np_val
            self.y_val = np_val[:, self.target_idx]
            self.val_dataset = ANNDataset(
                self.X_val, self.y_val, self.seq_len, self.pred_len, self.stride
            )

        if stage in (None, "test"):
            ## Training + validation dataset:
            np_train_val = pd.read_parquet(
                self.data_dir / "train_val_scaled.parquet"
            ).to_numpy()
            self.X_train_val = np_train_val
            self.y_train_val = np_train_val[:, self.target_idx]
            self.train_val_dataset = ANNDataset(
                self.X_train_val,
                self.y_train_val,
                self.seq_len,
                self.pred_len,
                self.stride,
            )

            ## Testing dataset:
            np_test = pd.read_parquet(self.data_dir / "test_scaled.parquet").to_numpy()
            self.X_test = np_test
            self.y_test = np_test[:, self.target_idx]
            self.test_dataset = ANNDataset(
                self.X_test, self.y_test, self.seq_len, self.pred_len, self.stride
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Real-time constraint
            num_workers=2,
            persistent_workers=False,  # False for HPC cluster
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Real-time constraint
            num_workers=2,
            persistent_workers=False,  # False for HPC cluster
        )

    def train_val_dataloader(self):
        return DataLoader(
            self.train_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Real-time constraint
            num_workers=2,
            persistent_workers=False,  # False for HPC cluster
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Real-time constraint
            num_workers=2,
            persistent_workers=False,  # False for HPC cluster
        )
