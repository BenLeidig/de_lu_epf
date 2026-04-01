from pathlib import Path

import pandas as pd
from joblib import load
from yaml import safe_load


def fetch_test_data(model_type: str):
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data/processed"
    df_test_scaled = pd.read_parquet(DATA_DIR / f"{model_type}/test_scaled.parquet")
    df_test = pd.read_parquet(DATA_DIR / f"{model_type}/test.parquet")
    return df_test_scaled, df_test


def fetch_features_targets(model_type: str):
    BASE_DIR = Path(__file__).parent.parent.parent
    CFG_DIR = BASE_DIR / "configs/models"
    with open(CFG_DIR / "preprocess_config.yaml") as f:
        cfg = safe_load(f)[model_type]
    return cfg["features"], cfg["targets"]


def fetch_fitted(model_type: str, model_name: str):
    BASE_DIR = Path(__file__).parent.parent.parent
    MODEL_DIR = BASE_DIR / f"models/{model_type}"
    return load(MODEL_DIR / model_name)


def fetch_full_scaler(model_type: str):
    BASE_DIR = Path(__file__).parent.parent.parent
    MODEL_DIR = BASE_DIR / f"models/{model_type}/full"
    return load(MODEL_DIR / "scaler.pkl")


def get_predictions_dmf(model_name: str):
    model_type = "dmf"

    df_test_scaled, _ = fetch_test_data(model_type)
    features, _ = fetch_features_targets(model_type)
    X_test_scaled = df_test_scaled[features]
    dt_index = X_test_scaled.index

    dmf = fetch_fitted(model_type=model_type, model_name=model_name)

    Y_test_pred = dmf.predict(X_test_scaled)
    y_test_pred = pd.DataFrame(
        data=Y_test_pred.to_numpy().flatten(), columns="price", index=dt_index
    )

    return y_test_pred
