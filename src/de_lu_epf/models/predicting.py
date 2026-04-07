from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from joblib import load
from yaml import safe_load

from de_lu_epf.data.loading import ANNDataModule
from de_lu_epf.models.architectures import TCN_LSTM_MHA
from de_lu_epf.models.training import get_best_vtlm_params


def fetch_train_val_data(model_type: str):
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data/processed"
    df_train_val_scaled = pd.read_parquet(
        DATA_DIR / f"{model_type}/train_val_scaled.parquet"
    )
    df_train_val = pd.read_parquet(DATA_DIR / f"{model_type}/train_val.parquet")
    return df_train_val_scaled, df_train_val


def fetch_test_data(model_type: str):
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data/processed"
    df_test_scaled = pd.read_parquet(DATA_DIR / f"{model_type}/test_scaled.parquet")
    df_test = pd.read_parquet(DATA_DIR / f"{model_type}/test.parquet")
    return df_test_scaled, df_test


def fetch_features_targets(model_type: str):
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    CFG_DIR = BASE_DIR / "configs/models"
    with open(CFG_DIR / "preprocess_config.yaml") as f:
        cfg = safe_load(f)[model_type]
    return cfg["features"], cfg["targets"]


def fetch_fitted(model_type: str, model_name: str):
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    MODEL_DIR = BASE_DIR / f"models/{model_type}/full"
    return load(MODEL_DIR / f"{model_name}.pkl")


def fetch_full_scalers(model_type: str):
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    MODEL_DIR = BASE_DIR / f"models/{model_type}/full"
    feature_scaler = load(MODEL_DIR / "feature_scaler.pkl")
    target_scaler = load(MODEL_DIR / "target_scaler.pkl")
    return feature_scaler, target_scaler


def format_preds(preds, which: str):
    if which == "dmf":
        return np.asarray(preds).flatten()
    elif which == "hybrid":
        return np.asarray(preds).sum(axis=1)


def get_predictions_dmf(model_name: str):
    model_type = "dmf"

    df_train_val_scaled, _ = fetch_train_val_data(model_type)
    df_test_scaled, _ = fetch_test_data(model_type)

    features, _ = fetch_features_targets(model_type)
    X_train_val_scaled = df_train_val_scaled[features]
    X_test_scaled = df_test_scaled[features]

    dmf = fetch_fitted(model_type=model_type, model_name=model_name)

    Y_train_val_pred = dmf.predict(X_train_val_scaled)
    Y_test_pred = dmf.predict(X_test_scaled)

    train_val_idx = pd.date_range(
        start=df_train_val_scaled.index[0] + pd.Timedelta(value=1, unit="h"),
        periods=Y_train_val_pred.shape[0] * Y_train_val_pred.shape[1],
        freq="h",
        tz="UTC",
    )
    test_idx = pd.date_range(
        start=df_test_scaled.index[0] + pd.Timedelta(value=1, unit="h"),
        periods=Y_test_pred.shape[0] * Y_test_pred.shape[1],
        freq="h",
        tz="UTC",
    )

    y_train_val_pred = pd.DataFrame(
        data=format_preds(preds=Y_train_val_pred, which="dmf"),
        columns=["price"],
        index=train_val_idx,
    )
    y_test_pred = pd.DataFrame(
        data=format_preds(preds=Y_test_pred, which="dmf"),
        columns=["price"],
        index=test_idx,
    )

    return y_train_val_pred, y_test_pred


def get_predictions_hybrid(model_name: str):

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data/processed/hybrid"
    CFG_DIR = BASE_DIR / "configs"
    MODEL_DIR = BASE_DIR / "models/hybrid/full"

    with open(CFG_DIR / "data/process_config.yaml") as f:
        dt_range_cfg = safe_load(f)["dt_range"]

    train_val_pred_dict = {}
    test_pred_dict = {}
    targets = ["imf1", "imf2", "imf3", "imf4", "imf5", "imf_resid"]
    for target_col in targets:
        batch_size, params = get_best_vtlm_params(target_col=target_col)

        datamodule = ANNDataModule(
            data_dir=DATA_DIR, batch_size=batch_size, target_col=target_col
        )
        datamodule.setup()
        train_val_dataloader = datamodule.train_val_dataloader()
        test_dataloader = datamodule.test_dataloader()

        state_dict = torch.load(
            MODEL_DIR / f"{model_name}/{target_col}_{model_name}.pt"
        )

        model = TCN_LSTM_MHA(input_size=datamodule.input_size, **params)
        model.load_state_dict(state_dict=state_dict)
        model.eval()

        trainer = pl.Trainer(
            accelerator="auto",
            logger=False,
            enable_checkpointing=False,
        )

        y_train_val_pred = trainer.predict(model, dataloaders=train_val_dataloader)
        train_val_pred_dict[target_col] = (
            torch.cat(y_train_val_pred, dim=0).detach().cpu().numpy()  # type: ignore
        )

        y_test_pred = trainer.predict(model, dataloaders=test_dataloader)
        test_pred_dict[target_col] = (
            torch.cat(y_test_pred, dim=0).detach().cpu().numpy()  # type: ignore
        )

    train_val_start = pd.to_datetime(
        dt_range_cfg["train"]["start"], utc=True
    ) + pd.Timedelta(24 * 7 * 4, "h")
    test_start = pd.to_datetime(dt_range_cfg["test"]["start"], utc=True) + pd.Timedelta(
        24 * 7 * 4, "h"
    )

    train_val_idx = pd.date_range(
        start=train_val_start,
        periods=len(train_val_pred_dict["imf1"]) * 24,
        freq="h",
    )

    test_idx = pd.date_range(
        start=test_start,
        periods=len(test_pred_dict["imf1"]) * 24,
        freq="h",
    )

    train_val_pred_df = pd.DataFrame(train_val_pred_dict)
    train_val_pred_df["datetime"] = train_val_idx
    train_val_pred_df = train_val_pred_df.set_index("datetime")
    train_val_pred_df = train_val_pred_df[train_val_pred_df.index.year < 2024]

    test_pred_df = pd.DataFrame(test_pred_dict)
    test_pred_df["datetime"] = test_idx
    test_pred_df = test_pred_df.set_index("datetime")
    test_pred_df = test_pred_df[test_pred_df.index.year == 2024]

    _, target_scaler = fetch_full_scalers(model_type="hybrid")
    train_val_pred_df = pd.DataFrame(
        data=target_scaler.inverse_transform(train_val_pred_df),
        columns=train_val_pred_df.columns,
        index=train_val_pred_df.index,
    )
    test_pred_df = pd.DataFrame(
        data=target_scaler.inverse_transform(test_pred_df),
        columns=test_pred_df.columns,
        index=test_pred_df.index,
    )

    train_val_pred_df["price"] = train_val_pred_df.sum(axis=1)
    test_pred_df["price"] = test_pred_df.sum(axis=1)

    return train_val_pred_df, test_pred_df
