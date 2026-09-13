from pathlib import Path
from typing import Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ArrayLike = Union[np.ndarray, torch.Tensor]


def _create_dmf_data(set: str, features: list, target: str):
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    data_path = BASE_DIR / "data/processed/dmf"
    df = pd.read_parquet(data_path / f"{set}_scaled.parquet")
    X = df[features]
    y = df[target]
    return X, y


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

    # Single source of truth for valid split names and the parquet file each one maps to.
    _SPLIT_FILES = {
        "train": "train_scaled.parquet",
        "val": "val_scaled.parquet",
        "train_val": "train_val_scaled.parquet",
        "test": "test_scaled.parquet",
    }

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
        self._datasets: dict[str, ANNDataset] = {}  # per-split lazy-load cache

    def setup(self, stage=None):
        # Must call ANNDataModule.setup() before calling train_dataloader() /
        # val_dataloader() / test_dataloader() / train_val_dataloader().
        # Delegates to the same lazy, cached per-split loader that
        # get_dataloader() uses, so a split read here (e.g. via stage="fit")
        # isn't re-read if get_dataloader() is later called for that split.
        self._ensure_schema()

        if stage in (
            None,
            "fit",
        ):  ## fit stage constructs train and validation datasets.
            self.train_dataset = self._load_dataset("train")
            self.val_dataset = self._load_dataset("val")

        if stage in (None, "test"):
            self.train_val_dataset = self._load_dataset("train_val")
            self.test_dataset = self._load_dataset("test")

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("val")

    def train_val_dataloader(self):
        return self.get_dataloader("train_val")

    def test_dataloader(self):
        return self.get_dataloader("test")

    def _ensure_schema(self) -> None:
        # Reads train_scaled.parquet once to derive target_idx / input_size,
        # independent of setup() and however get_dataloader() is called.
        if getattr(self, "target_idx", None) is None:
            df_train = pd.read_parquet(self.data_dir / self._SPLIT_FILES["train"])
            self.target_idx = df_train.columns.get_loc(self.target_col)
            self.input_size = df_train.shape[1]

    def _load_dataset(self, split: str) -> ANNDataset:
        """Lazily load (and cache) the ANNDataset for a single named split.

        Args:
            split (str): One of "train", "val", "train_val", "test" (case- and
                whitespace-insensitive).

        Raises:
            ValueError: If ``split`` doesn't match one of the known splits.
        """
        key = split.strip().lower()
        if key not in self._SPLIT_FILES:
            raise ValueError(
                f"Unknown split {split!r}. Valid options: {sorted(self._SPLIT_FILES)}"
            )

        if key not in self._datasets:
            self._ensure_schema()
            np_arr = pd.read_parquet(self.data_dir / self._SPLIT_FILES[key]).to_numpy()
            X = np_arr
            y = np_arr[:, self.target_idx]
            self._datasets[key] = ANNDataset(
                X, y, self.seq_len, self.pred_len, self.stride
            )

        return self._datasets[key]

    def get_dataloader(self, split: str) -> DataLoader:
        """Get a DataLoader for a single named split, loading only that split.

        Unlike ``setup(stage=None)`` (which eagerly reads all four splits),
        this loads at most the one parquet file requested, and caches it per
        instance so repeated calls (e.g. the same split used for both a
        "train" and "test" context) don't re-read the file. Intended for
        callers that need to pick an arbitrary split by name at runtime, such
        as generating predictions over a caller-chosen train/test split.

        Args:
            split (str): One of "train", "val", "train_val", "test" (case- and
                whitespace-insensitive).

        Raises:
            ValueError: If ``split`` doesn't match one of the known splits.
        """
        dataset = self._load_dataset(split)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Real-time constraint
            num_workers=2,
            persistent_workers=False,  # False for HPC cluster
        )
