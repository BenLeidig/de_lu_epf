import gc
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import optuna
import torch

from src.data.datamodule import ANNDataModule
from src.models.components.tcn_lstm_mha import TCN_LSTM_MHA


def tune_vmd_tcn_lstm_mha(
    target_col: str,
    seq_len: int,
    pred_len: int,
    stride: int,
    batch_size_range: list,
    lr_init_range: list,
    kernel_size_range: list,
    tcn_dropout_range: list,
    lstm_dropouts_range: list,
    mha_dropout_range: list,
    mha_heads_range: list,
    patience: int = 5,
    max_epochs: int = 50,
    reduction_factor: int = 3,
    n_trials: int = 100,
    accelerator: str = "gpu",
    random_state: int = 0,
):
    pl.seed_everything(random_state)

    def objective(trial: optuna.trial.Trial):

        #### general params ####
        batch_size = trial.suggest_int(
            "batch_size", batch_size_range[0], batch_size_range[1], log=True
        )
        lr_init = trial.suggest_float(
            "lr_init", lr_init_range[0], lr_init_range[1], log=True
        )

        #### TCN params ####
        dilation_base = 2
        kernel_size = trial.suggest_int(
            "kernel_size", kernel_size_range[0], kernel_size_range[1]
        )
        num_blocks = int(
            np.ceil(
                np.log2(
                    ((seq_len - 1) * (dilation_base - 1)) / ((kernel_size - 1) * 2) + 1
                )
            )
        )

        channel_sizes = list(np.zeros(num_blocks, dtype=int))
        for i in range(num_blocks):
            channel_sizes[i] = trial.suggest_int(
                f"channel_size_{num_blocks}_{i}",
                channel_sizes[i - 1] if i > 0 else 8,
                256,
            )

        tcn_dropout = trial.suggest_float(
            "tcn_dropout", tcn_dropout_range[0], tcn_dropout_range[1]
        )

        #### LSTM params ####
        hidden_size0 = trial.suggest_int(
            "hidden_size0", max(16, channel_sizes[-1] // 4), min(256, channel_sizes[-1])
        )
        hidden_size1 = trial.suggest_int(
            "hidden_size1", 16, min(256, channel_sizes[-1])
        )
        hidden_size2 = trial.suggest_int(
            "hidden_size2", 16, min(256, channel_sizes[-1])
        )
        hidden_sizes = [hidden_size0, hidden_size1, hidden_size2]

        lstm_dropout0 = trial.suggest_float(
            "lstm_dropout0", lstm_dropouts_range[0], lstm_dropouts_range[1]
        )
        lstm_dropout1 = trial.suggest_float(
            "lstm_dropout1", lstm_dropouts_range[0], lstm_dropouts_range[1]
        )
        lstm_dropouts = [lstm_dropout0, lstm_dropout1]

        #### MHA params ####
        mha_dropout = trial.suggest_float(
            "mha_dropout", mha_dropout_range[0], mha_dropout_range[1]
        )
        mha_heads = trial.suggest_categorical("mha_heads", mha_heads_range)
        if hidden_sizes[-1] % mha_heads != 0:  # type: ignore
            raise optuna.TrialPruned()

        #### callbacks ####
        callbacks = [
            pl.callbacks.EarlyStopping(  # type: ignore
                monitor="val_loss", patience=patience, mode="min"
            ),
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val_loss"
            ),
        ]

        #### training ####
        ## making the dataset considering the batch_size
        datamodule = ANNDataModule(
            data_dir=Path(__file__).resolve().parent.parent.parent.parent.parent
            / "data/processed/vmd_data",
            batch_size=batch_size,
            target_col=target_col,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
        )
        datamodule.setup("fit")
        input_size = datamodule.input_size

        mod = TCN_LSTM_MHA(  ## instantiating the model
            input_size=input_size,
            channel_sizes=channel_sizes,
            kernel_size=kernel_size,
            hidden_sizes=hidden_sizes,
            tcn_dropout=tcn_dropout,
            lstm_dropouts=lstm_dropouts,
            mha_dropout=mha_dropout,
            mha_heads=mha_heads,  # type: ignore
            lr_init=lr_init,
        )

        trainer = (
            pl.Trainer(  ## instantiating the trainer given the model and callbacks
                max_epochs=max_epochs,
                callbacks=callbacks,
                accelerator=accelerator,
                logger=False,
                enable_checkpointing=False,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
        )
        trainer.fit(mod, datamodule=datamodule)  ## fitting the trainer
        val_loss = trainer.callback_metrics["val_loss"].item()  ## finding the loss

        del trainer, mod, datamodule
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return val_loss  ## report the loss

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=patience,
        max_resource=max_epochs,
        reduction_factor=reduction_factor,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    return study
