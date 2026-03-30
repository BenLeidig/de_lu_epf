from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd

from src.data.datamodule import ANNDataModule
from src.models.components.tcn_lstm_mha import TCN_LSTM_MHA


def get_vtlm_params(best_params: dict):
    batch_size = best_params.pop("batch_size")
    hidden_sizes = [
        best_params.pop("hidden_size0"),
        best_params.pop("hidden_size1"),
        best_params.pop("hidden_size2"),
    ]
    lstm_dropouts = [best_params.pop("lstm_dropout0"), best_params.pop("lstm_dropout1")]
    channel_sizes = []
    channel_size_keys = list(best_params.keys())
    for s in channel_size_keys:
        if "channel_size_" in s:
            channel_sizes.append(best_params.pop(s))
    params = best_params.copy()
    params["hidden_sizes"] = hidden_sizes
    params["lstm_dropouts"] = lstm_dropouts
    params["channel_sizes"] = channel_sizes
    return batch_size, params


def get_ann_fitted(
    target_col: str,
    batch_size: int,
    params: dict,
    data_dir: Path,
    seq_len: int,
    pred_len: int,
    stride: int,
    model_class,
    max_epochs: int = 80,
    accelerator="auto",
):

    datamodule = ANNDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        target_col=target_col,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
    )
    datamodule.setup("test")
    train_val_dataloader = datamodule.train_val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    input_size = datamodule.input_size

    model = model_class(input_size=input_size, **params)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_dataloaders=train_val_dataloader)
    trainer.predict(model, dataloaders=test_dataloader)

    return batch_size, model, trainer
