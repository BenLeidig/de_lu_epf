from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from yaml import safe_load

from de_lu_epf.data.loading import ANNDataModule
from de_lu_epf.models.architectures import DirectMultiStepForecaster
from de_lu_epf.models.hpo.dmf_tuning import supports_parallel, supports_random_state

## NOTE: a lot of this code is fairly readable and straightforward.
## Many code comments are not included here for brevity and to improve
## readability. It should be noted, however, that date handling (in
## this source code) is set explicity for this research project.
## Reach out to ben.leidig@gmail.com if you have any questions.


def get_fitted_dmf(model_name: str, final: bool = True):
    """Fit a DMF (classical ML) model.

    Args:
        model_name (str): Which model class to fit (e.g. "en", "svr", "rfr").
        final (bool): If True (default), fit on the combined `train_val`
            split. This is the single, already-selected winning model's
            final refit, done once before its one-time `test` evaluation.
            If False, fit on `train` only, holding `val` out, producing a
            "candidate" model whose validation-split performance can be
            used to compare models before a winner is selected.
    """

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    CFG_DIR = BASE_DIR / "configs/models"
    DATA_DIR = BASE_DIR / "data/processed/dmf"

    with open(CFG_DIR / "preprocess_config.yaml") as f:
        cfg = safe_load(f)["dmf"]
    features = cfg["features"]
    targets = cfg["targets"]

    split = "train_val" if final else "train"
    df_train_val = pd.read_parquet(DATA_DIR / f"{split}_scaled.parquet")
    X_train_val = df_train_val[features]
    Y_train_val = df_train_val[targets]

    model_classes = {
        "lr": LinearRegression,
        "en": ElasticNet,
        "svr": SVR,
        "rfr": RandomForestRegressor,
        "lgbmr": LGBMRegressor,
        "xgbr": XGBRegressor,
    }

    model_class = model_classes[model_name]

    if model_name != "lr":
        with open(CFG_DIR / "dmf_hyperparams_config.yaml") as f:
            params = safe_load(f)[model_name]
    else:
        params = {i: {} for i in range(0, 24)}

    if supports_random_state(model_class):
        for i in range(0, 24):
            params[i]["random_state"] = 0
    if supports_parallel(model_class):
        for i in range(0, 24):
            params[i]["n_jobs"] = -1

    dmf = DirectMultiStepForecaster(params=params, model_class=model_class)
    dmf.fit(X_train_val, Y_train_val)

    return dmf


def get_best_ann_params(target_col: str, model_name: str, model_type: str):

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    CFG_DIR = BASE_DIR / "configs/models"

    with open(CFG_DIR / f"{model_type}_hyperparams_config.yaml") as f:
        if model_type == "hybrid":
            best_params = safe_load(f)[model_name.replace("_", "-")][target_col].copy()
        else:
            best_params = safe_load(f)[model_name.replace("_", "-")].copy()

    batch_size = best_params.pop("batch_size")
    params = {"lr_init": best_params.pop("lr_init")}

    if "hidden_size0" in best_params:
        hidden_sizes = [
            best_params.pop("hidden_size0"),
            best_params.pop("hidden_size1"),
            best_params.pop("hidden_size2"),
        ]
        lstm_dropouts = [
            best_params.pop("lstm_dropout0"),
            best_params.pop("lstm_dropout1"),
        ]
        params["hidden_sizes"] = hidden_sizes
        params["lstm_dropouts"] = lstm_dropouts

    if any("channel_size" in key for key in best_params):
        channel_sizes = []
        channel_size_keys = sorted(
            [key for key in best_params if "channel_size" in key],
            key=lambda x: int(x.split("_")[-1]),
        )
        for s in channel_size_keys:
            channel_sizes.append(best_params.pop(s))
        params["channel_sizes"] = channel_sizes
        params["tcn_dropout"] = best_params.pop("tcn_dropout")
        params["kernel_size"] = best_params.pop("kernel_size")

    if "mha_dropout" in best_params:
        params["mha_dropout"] = best_params.pop("mha_dropout")
        params["mha_heads"] = best_params.pop("mha_heads")

    return batch_size, params


def get_best_epoch(ckpt_path: Path) -> int:
    """Read the number of completed training epochs from a Lightning checkpoint.

    Used to carry a candidate model's val-monitored training length over to
    its final retrain on train+val, where there's no held-out val split left
    to early-stop against (see `get_fitted_ann(..., final=True)`).

    Args:
        ckpt_path (Path): Path to a `.ckpt` file written by `get_fitted_ann`
            with `final=False` (i.e. a candidate model's best checkpoint).

    Returns:
        int: The number of epochs completed up to and including the
            checkpointed (best val_loss) epoch.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return checkpoint["epoch"] + 1  # epoch is 0-indexed; +1 gives a count


def get_fitted_ann(
    model_class,
    params: dict,
    target_col: str,
    model_type: str,
    model_name: str,
    seq_len: int,
    pred_len: int,
    stride: int,
    batch_size: int,
    patience: int = 5,
    max_epochs: int = 50,
    accelerator: str = "gpu",
    random_state: int = 0,
    final: bool = False,
    num_epochs: Optional[int] = None,
):
    """Fit an ANN/hybrid model.

    Args:
        final (bool): If False (default), fit on `train`, early-stopping and
            checkpointing on `val`. Produces a "candidate" model whose
            validation-split performance can be used to compare
            architectures before a winner is selected. If True, this is the
            single, already-selected winning model: it is refit on the
            combined `train_val` split for exactly `num_epochs` epochs (no
            held-out val remains to monitor), producing the final model that
            gets evaluated once on `test`.
        num_epochs (Optional[int]): Required when `final=True`. The number
            of epochs to train for, normally read off the winning
            candidate's best checkpoint via `get_best_epoch()`. Ignored when
            `final=False`.
    """
    pl.seed_everything(random_state)

    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    # NOTE: "full" holds candidate (train/val-fitted) ANN/hybrid checkpoints;
    ## "final" holds the single winning model's train_val-refit checkpoint.
    ## (For DMF, by contrast, "full" already means "final" - see
    ## get_fitted_dmf - since it was the only mode that existed before this
    ## comparison/final split was introduced. Kept as-is to avoid moving any
    ## existing DMF artifacts.)
    stage_dir = "final" if final else "full"
    dirpath = (
        BASE_DIR / f"models/{model_type}/{stage_dir}/{model_name}"
        if model_type == "hybrid"
        else BASE_DIR / f"models/{model_type}/{stage_dir}"
    )

    datamodule = ANNDataModule(
        data_dir=BASE_DIR / f"data/processed/{model_type}",
        batch_size=batch_size,
        target_col=target_col,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
    )

    if final:
        if num_epochs is None:
            raise ValueError(
                "num_epochs is required when final=True: the final retrain on "
                "train_val has no held-out val split to early-stop against, so "
                "it must train for a fixed epoch count (typically the winning "
                "candidate's best-epoch count from get_best_epoch())."
            )

        datamodule.setup("test")  # loads train_val (+ test, unused here)
        input_size = datamodule.input_size
        mod = model_class(input_size=input_size, **params)

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator=accelerator,
            logger=False,
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )
        trainer.fit(mod, train_dataloaders=datamodule.train_val_dataloader())

        dirpath.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(dirpath / f"{target_col}_{model_name}.ckpt")
    else:
        early_stopping_cb = pl.callbacks.EarlyStopping(  # type: ignore
            monitor="val_loss", patience=patience, mode="min"
        )
        ckpt_cb = pl.callbacks.ModelCheckpoint(  # type: ignore
            dirpath=dirpath,
            filename=f"{target_col}_{model_name}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
        )

        datamodule.setup("fit")
        input_size = datamodule.input_size
        mod = model_class(input_size=input_size, **params)

        trainer = pl.Trainer(  ## instantiating the trainer given the model and callbacks
            max_epochs=max_epochs,
            callbacks=[early_stopping_cb, ckpt_cb],
            accelerator=accelerator,
            logger=False,
            enable_checkpointing=True,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )
        trainer.fit(mod, datamodule=datamodule)  ## fitting the trainer
