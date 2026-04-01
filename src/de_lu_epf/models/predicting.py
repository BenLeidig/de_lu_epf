from pathlib import Path

import pandas as pd
from joblib import load
from yaml import safe_load


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


def fetch_full_scaler(model_type: str):
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    MODEL_DIR = BASE_DIR / f"models/{model_type}/full"
    return load(MODEL_DIR / "scaler.pkl")


def format_dmf_preds(preds):
    return preds.to_numpy().flatten()


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
        start=df_train_val_scaled.index[0],
        periods=Y_train_val_pred.shape[0] * Y_train_val_pred.shape[1],
        freq="h",
        tz="UTC",
    )
    test_idx = pd.date_range(
        start=df_test_scaled.index[0],
        periods=Y_test_pred.shape[0] * Y_test_pred.shape[1],
        freq="h",
        tz="UTC",
    )

    y_train_val_pred = pd.DataFrame(
        data=format_dmf_preds(Y_train_val_pred), columns=["price"], index=train_val_idx
    )
    y_test_pred = pd.DataFrame(
        data=format_dmf_preds(Y_test_pred), columns=["price"], index=test_idx
    )

    return y_train_val_pred, y_test_pred
