from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
from torch import lstm
from xgboost import XGBRegressor
from yaml import safe_load

from de_lu_epf.data.loading import ANNDataModule
from de_lu_epf.models.architectures import DirectMultiStepForecaster
from de_lu_epf.models.hpo.dmf_tuning import supports_parallel, supports_random_state


def get_fitted_dmf(model_name: str):

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    CFG_DIR = BASE_DIR / "configs/models"
    DATA_DIR = BASE_DIR / "data/processed/dmf"

    with open(CFG_DIR / "preprocess_config.yaml") as f:
        cfg = safe_load(f)["dmf"]
    features = cfg["features"]
    targets = cfg["targets"]

    df_train_val = pd.read_parquet(DATA_DIR / "train_val_scaled.parquet")
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
            best_params = safe_load(f)[model_name][target_col].copy()
        else:
            best_params = safe_load(f)[model_name].copy()

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
):
    pl.seed_everything(random_state)

    early_stopping_cb = pl.callbacks.EarlyStopping(  # type: ignore
        monitor="val_loss", patience=patience, mode="min"
    )
    ckpt_cb = pl.callbacks.ModelCheckpoint(  # type: ignore
        dirpath=Path(__file__).resolve().parent.parent.parent.parent.parent
        / f"models/{model_type}/full/{model_name}",
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    callbacks = [early_stopping_cb, ckpt_cb]

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

    mod = model_class(input_size=input_size, **params)

    trainer = pl.Trainer(  ## instantiating the trainer given the model and callbacks
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(mod, datamodule=datamodule)  ## fitting the trainer
