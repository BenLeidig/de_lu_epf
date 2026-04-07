from pathlib import Path

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


def get_best_vtlm_params(target_col: str):

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    CFG_DIR = BASE_DIR / "configs/models"

    with open(CFG_DIR / "hybrid_hyperparams_config.yaml") as f:
        best_params = safe_load(f)["vtlm"][target_col]

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


def get_fitted_ann(
    target_col: str,
    batch_size: int,
    params: dict,
    data_dir: Path,
    model_class,
    seq_len: int = 24 * 7 * 4,
    pred_len: int = 24,
    stride: int = 24,
    max_epochs: int = 35,
    accelerator="auto",
):

    pl.seed_everything(0)

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
    # test_dataloader = datamodule.test_dataloader()
    input_size = datamodule.input_size

    model = model_class(input_size=input_size, **params)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_dataloaders=train_val_dataloader)
    # y_test_pred = trainer.predict(model, dataloaders=test_dataloader)
    # y_test_pred = torch.cat(y_test_pred, dim=0)  # type: ignore

    return model  # , y_test_pred
