import gc
import shutil
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import optuna
import torch

from de_lu_epf.data.loading import ANNDataModule
from de_lu_epf.models.architectures import (
    LSTM,
    LSTM_MHA,
    TCN,
    TCN_LSTM,
    TCN_LSTM_MHA,
    TCN_MHA,
)


def _fit_and_checkpoint(
    trial: optuna.trial.Trial,
    mod: pl.LightningModule,
    datamodule: ANNDataModule,
    patience: int,
    max_epochs: int,
    accelerator: str,
    checkpoint_path: Path,
) -> float:
    """Fit one HPO trial, and persist its best-epoch weights as the new
    best-so-far checkpoint for this study, if this trial improves on every
    trial completed before it.

    This lets HPO's own winning trial be used directly as the "candidate"
    model for cross-architecture comparison (see select_final_model.py),
    instead of a separate script re-fitting the winning hyperparameters
    from scratch afterward.

    Every trial writes to the same scratch directory, cleared at the end of
    each trial. (Lightning's ModelCheckpoint auto-versions the filename -
    "scratch-v1.ckpt", "-v2.ckpt", etc. - if a file already exists at that
    path when a new Trainer starts, rather than overwriting it, so simply
    reusing a fixed path is not enough on its own; the directory has to be
    emptied explicitly between trials.) This keeps a long search from
    accumulating leftover checkpoint files from trials that didn't win, or
    were pruned partway through.

    Returns:
        float: This trial's *best*-epoch val_loss (what the checkpoint, if
            kept, actually achieves) - not the last-epoch val_loss, which
            early stopping's patience window typically leaves worse.
    """
    tmp_dir = checkpoint_path.parent / ".tmp" / checkpoint_path.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = pl.callbacks.ModelCheckpoint(  # type: ignore
        dirpath=tmp_dir,
        filename="scratch",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    callbacks = [
        pl.callbacks.EarlyStopping(  # type: ignore
            monitor="val_loss", patience=patience, mode="min"
        ),
        optuna.integration.PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ckpt_cb,
    ]

    trainer = pl.Trainer(  ## instantiating the trainer given the model and callbacks
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    try:
        trainer.fit(mod, datamodule=datamodule)  ## fitting the trainer
        # trainer.fit() raises optuna.TrialPruned() from inside the pruning
        # callback for a pruned trial, so nothing below this line runs for
        # one - which is correct: a pruned trial should never be compared
        # against/promoted over the best completed trial so far. The
        # `finally` block below still runs the scratch-directory cleanup
        # either way.

        # The checkpoint's own best score, not trainer.callback_metrics
        # ["val_loss"] (which reflects the LAST epoch trained - typically
        # several epochs worse than the best one, due to early stopping's
        # patience window).
        val_loss = ckpt_cb.best_model_score.item()  ## finding the loss

        try:
            is_best = val_loss < trial.study.best_value
        except ValueError:  # no trial has completed yet - this one is best by default
            is_best = True

        if is_best and ckpt_cb.best_model_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ckpt_cb.best_model_path, checkpoint_path)
    finally:
        # Always clear the scratch directory - completed, pruned, or errored
        # - so the next trial starts from a clean slate and reuses the plain
        # "scratch.ckpt" name, rather than Lightning auto-versioning around
        # whatever this trial left behind.
        shutil.rmtree(tmp_dir, ignore_errors=True)

        del trainer, mod, datamodule
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return val_loss  ## report the loss


def tune_tcn_lstm_mha(
    target_col: str,
    model_type: str,
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
    checkpoint_path: Path,
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
            "batch_size", int(batch_size_range[0]), int(batch_size_range[1]), log=True
        )
        lr_init = trial.suggest_float(
            "lr_init", float(lr_init_range[0]), float(lr_init_range[1]), log=True
        )

        #### TCN params ####
        dilation_base = 2
        kernel_size = trial.suggest_int(
            "kernel_size", int(kernel_size_range[0]), int(kernel_size_range[1])
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
            "tcn_dropout", float(tcn_dropout_range[0]), float(tcn_dropout_range[1])
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
            "lstm_dropout0",
            float(lstm_dropouts_range[0]),
            float(lstm_dropouts_range[1]),
        )
        lstm_dropout1 = trial.suggest_float(
            "lstm_dropout1",
            float(lstm_dropouts_range[0]),
            float(lstm_dropouts_range[1]),
        )
        lstm_dropouts = [lstm_dropout0, lstm_dropout1]

        #### MHA params ####
        mha_dropout = trial.suggest_float(
            "mha_dropout", float(mha_dropout_range[0]), float(mha_dropout_range[1])
        )
        mha_heads = trial.suggest_categorical(
            "mha_heads", [int(h) for h in mha_heads_range]
        )
        if hidden_sizes[-1] % mha_heads != 0:  # type: ignore
            raise optuna.TrialPruned()

        #### training ####
        ## making the dataset considering the batch_size
        datamodule = ANNDataModule(
            data_dir=Path(__file__).resolve().parent.parent.parent.parent.parent
            / f"data/processed/{model_type}",
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

        return _fit_and_checkpoint(
            trial, mod, datamodule, patience, max_epochs, accelerator, checkpoint_path
        )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=patience,
        max_resource=max_epochs,
        reduction_factor=reduction_factor,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    return study


def tune_tcn_lstm(
    target_col: str,
    model_type: str,
    seq_len: int,
    pred_len: int,
    stride: int,
    batch_size_range: list,
    lr_init_range: list,
    kernel_size_range: list,
    tcn_dropout_range: list,
    lstm_dropouts_range: list,
    checkpoint_path: Path,
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
            "batch_size", int(batch_size_range[0]), int(batch_size_range[1]), log=True
        )
        lr_init = trial.suggest_float(
            "lr_init", float(lr_init_range[0]), float(lr_init_range[1]), log=True
        )

        #### TCN params ####
        dilation_base = 2
        kernel_size = trial.suggest_int(
            "kernel_size", int(kernel_size_range[0]), int(kernel_size_range[1])
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
            "tcn_dropout", float(tcn_dropout_range[0]), float(tcn_dropout_range[1])
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
            "lstm_dropout0",
            float(lstm_dropouts_range[0]),
            float(lstm_dropouts_range[1]),
        )
        lstm_dropout1 = trial.suggest_float(
            "lstm_dropout1",
            float(lstm_dropouts_range[0]),
            float(lstm_dropouts_range[1]),
        )
        lstm_dropouts = [lstm_dropout0, lstm_dropout1]

        #### training ####
        ## making the dataset considering the batch_size
        datamodule = ANNDataModule(
            data_dir=Path(__file__).resolve().parent.parent.parent.parent.parent
            / f"data/processed/{model_type}",
            batch_size=batch_size,
            target_col=target_col,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
        )
        datamodule.setup("fit")
        input_size = datamodule.input_size

        mod = TCN_LSTM(  ## instantiating the model
            input_size=input_size,
            channel_sizes=channel_sizes,
            kernel_size=kernel_size,
            hidden_sizes=hidden_sizes,
            tcn_dropout=tcn_dropout,
            lstm_dropouts=lstm_dropouts,
            lr_init=lr_init,
        )

        return _fit_and_checkpoint(
            trial, mod, datamodule, patience, max_epochs, accelerator, checkpoint_path
        )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=patience,
        max_resource=max_epochs,
        reduction_factor=reduction_factor,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    return study


def tune_tcn_mha(
    target_col: str,
    model_type: str,
    seq_len: int,
    pred_len: int,
    stride: int,
    batch_size_range: list,
    lr_init_range: list,
    kernel_size_range: list,
    tcn_dropout_range: list,
    mha_dropout_range: list,
    mha_heads_range: list,
    checkpoint_path: Path,
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
            "batch_size", int(batch_size_range[0]), int(batch_size_range[1]), log=True
        )
        lr_init = trial.suggest_float(
            "lr_init", float(lr_init_range[0]), float(lr_init_range[1]), log=True
        )

        #### TCN params ####
        dilation_base = 2
        kernel_size = trial.suggest_int(
            "kernel_size", int(kernel_size_range[0]), int(kernel_size_range[1])
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
            "tcn_dropout", float(tcn_dropout_range[0]), float(tcn_dropout_range[1])
        )

        #### MHA params ####
        mha_dropout = trial.suggest_float(
            "mha_dropout", float(mha_dropout_range[0]), float(mha_dropout_range[1])
        )
        mha_heads = trial.suggest_categorical(
            "mha_heads", [int(h) for h in mha_heads_range]
        )
        if channel_sizes[-1] % mha_heads != 0:  # type: ignore
            raise optuna.TrialPruned()

        #### training ####
        ## making the dataset considering the batch_size
        datamodule = ANNDataModule(
            data_dir=Path(__file__).resolve().parent.parent.parent.parent.parent
            / f"data/processed/{model_type}",
            batch_size=batch_size,
            target_col=target_col,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
        )
        datamodule.setup("fit")
        input_size = datamodule.input_size

        mod = TCN_MHA(  ## instantiating the model
            input_size=input_size,
            channel_sizes=channel_sizes,
            kernel_size=kernel_size,
            tcn_dropout=tcn_dropout,
            mha_dropout=mha_dropout,
            mha_heads=mha_heads,  # type: ignore
            lr_init=lr_init,
        )

        return _fit_and_checkpoint(
            trial, mod, datamodule, patience, max_epochs, accelerator, checkpoint_path
        )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=patience,
        max_resource=max_epochs,
        reduction_factor=reduction_factor,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    return study


def tune_tcn(
    target_col: str,
    model_type: str,
    seq_len: int,
    pred_len: int,
    stride: int,
    batch_size_range: list,
    lr_init_range: list,
    kernel_size_range: list,
    tcn_dropout_range: list,
    checkpoint_path: Path,
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
            "batch_size", int(batch_size_range[0]), int(batch_size_range[1]), log=True
        )
        lr_init = trial.suggest_float(
            "lr_init", float(lr_init_range[0]), float(lr_init_range[1]), log=True
        )

        #### TCN params ####
        dilation_base = 2
        kernel_size = trial.suggest_int(
            "kernel_size", int(kernel_size_range[0]), int(kernel_size_range[1])
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
            "tcn_dropout", float(tcn_dropout_range[0]), float(tcn_dropout_range[1])
        )

        #### training ####
        ## making the dataset considering the batch_size
        datamodule = ANNDataModule(
            data_dir=Path(__file__).resolve().parent.parent.parent.parent.parent
            / f"data/processed/{model_type}",
            batch_size=batch_size,
            target_col=target_col,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
        )
        datamodule.setup("fit")
        input_size = datamodule.input_size

        mod = TCN(  ## instantiating the model
            input_size=input_size,
            channel_sizes=channel_sizes,
            kernel_size=kernel_size,
            tcn_dropout=tcn_dropout,
            lr_init=lr_init,
        )

        return _fit_and_checkpoint(
            trial, mod, datamodule, patience, max_epochs, accelerator, checkpoint_path
        )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=patience,
        max_resource=max_epochs,
        reduction_factor=reduction_factor,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    return study


def tune_lstm(
    target_col: str,
    model_type: str,
    seq_len: int,
    pred_len: int,
    stride: int,
    batch_size_range: list,
    lr_init_range: list,
    lstm_dropouts_range: list,
    checkpoint_path: Path,
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
            "batch_size", int(batch_size_range[0]), int(batch_size_range[1]), log=True
        )
        lr_init = trial.suggest_float(
            "lr_init", float(lr_init_range[0]), float(lr_init_range[1]), log=True
        )

        #### LSTM params ####
        hidden_size0 = trial.suggest_int("hidden_size0", 16, 256)
        hidden_size1 = trial.suggest_int("hidden_size1", 16, 256)
        hidden_size2 = trial.suggest_int("hidden_size2", 16, 256)
        hidden_sizes = [hidden_size0, hidden_size1, hidden_size2]

        lstm_dropout0 = trial.suggest_float(
            "lstm_dropout0",
            float(lstm_dropouts_range[0]),
            float(lstm_dropouts_range[1]),
        )
        lstm_dropout1 = trial.suggest_float(
            "lstm_dropout1",
            float(lstm_dropouts_range[0]),
            float(lstm_dropouts_range[1]),
        )
        lstm_dropouts = [lstm_dropout0, lstm_dropout1]

        #### training ####
        ## making the dataset considering the batch_size
        datamodule = ANNDataModule(
            data_dir=Path(__file__).resolve().parent.parent.parent.parent.parent
            / f"data/processed/{model_type}",
            batch_size=batch_size,
            target_col=target_col,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
        )
        datamodule.setup("fit")
        input_size = datamodule.input_size

        mod = LSTM(  ## instantiating the model
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            lstm_dropouts=lstm_dropouts,
            lr_init=lr_init,
        )

        return _fit_and_checkpoint(
            trial, mod, datamodule, patience, max_epochs, accelerator, checkpoint_path
        )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=patience,
        max_resource=max_epochs,
        reduction_factor=reduction_factor,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    return study


def tune_lstm_mha(
    target_col: str,
    model_type: str,
    seq_len: int,
    pred_len: int,
    stride: int,
    batch_size_range: list,
    lr_init_range: list,
    lstm_dropouts_range: list,
    mha_dropout_range: list,
    mha_heads_range: list,
    checkpoint_path: Path,
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
            "batch_size", int(batch_size_range[0]), int(batch_size_range[1]), log=True
        )
        lr_init = trial.suggest_float(
            "lr_init", float(lr_init_range[0]), float(lr_init_range[1]), log=True
        )

        #### LSTM params ####
        hidden_size0 = trial.suggest_int("hidden_size0", 16, 256)
        hidden_size1 = trial.suggest_int("hidden_size1", 16, 256)
        hidden_size2 = trial.suggest_int("hidden_size2", 16, 256)
        hidden_sizes = [hidden_size0, hidden_size1, hidden_size2]

        lstm_dropout0 = trial.suggest_float(
            "lstm_dropout0",
            float(lstm_dropouts_range[0]),
            float(lstm_dropouts_range[1]),
        )
        lstm_dropout1 = trial.suggest_float(
            "lstm_dropout1",
            float(lstm_dropouts_range[0]),
            float(lstm_dropouts_range[1]),
        )
        lstm_dropouts = [lstm_dropout0, lstm_dropout1]

        #### MHA params ####
        mha_dropout = trial.suggest_float(
            "mha_dropout", float(mha_dropout_range[0]), float(mha_dropout_range[1])
        )
        mha_heads = trial.suggest_categorical(
            "mha_heads", [int(h) for h in mha_heads_range]
        )
        if hidden_sizes[-1] % mha_heads != 0:  # type: ignore
            raise optuna.TrialPruned()

        #### training ####
        ## making the dataset considering the batch_size
        datamodule = ANNDataModule(
            data_dir=Path(__file__).resolve().parent.parent.parent.parent.parent
            / f"data/processed/{model_type}",
            batch_size=batch_size,
            target_col=target_col,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
        )
        datamodule.setup("fit")
        input_size = datamodule.input_size

        mod = LSTM_MHA(  ## instantiating the model
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            lstm_dropouts=lstm_dropouts,
            mha_dropout=mha_dropout,
            mha_heads=mha_heads,  # type: ignore
            lr_init=lr_init,
        )

        return _fit_and_checkpoint(
            trial, mod, datamodule, patience, max_epochs, accelerator, checkpoint_path
        )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=patience,
        max_resource=max_epochs,
        reduction_factor=reduction_factor,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    return study
