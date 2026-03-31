from pathlib import Path

import pandas as pd
from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from yaml import safe_load

from src.models.architectures import DirectMultiStepForecaster
from src.models.hpo.dmf_tuning import supports_parallel, supports_random_state

if __name__ == "__main__":
    model_name = "lgbmr"

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    CONFIG_DIR = BASE_DIR / "configs/models"
    DATA_DIR = BASE_DIR / "data/processed/dmf_data"
    MODEL_DIR = BASE_DIR / "models/dmf_models/full"

    with open(CONFIG_DIR / "preprocess_config.yaml") as f:
        cfg = safe_load(f)["dmf"]
    features = cfg["features"]
    targets = cfg["targets"]

    df_train_val = pd.read_parquet(DATA_DIR / "train_val_scaled.parquet")
    X_train_val = df_train_val[features]
    Y_train_val = df_train_val[targets]

    model_classes = {
        "en": ElasticNet,
        "svr": SVR,
        "rfr": RandomForestRegressor,
        "lgbmr": LGBMRegressor,
        "xgbr": XGBRegressor,
    }

    model_class = model_classes[model_name]

    with open(CONFIG_DIR / "dmf_hyperparams_config.yaml") as f:
        params = safe_load(f)[model_name]
    if supports_random_state(model_class):
        params["random_state"] = 0
    if supports_parallel(model_class):
        params["n_jobs"] = -1

    dmf = DirectMultiStepForecaster(params=params, model_class=model_class)
    dmf.fit(X_train_val, Y_train_val)

    dump(dmf, MODEL_DIR / f"{model_name}_dmf.pkl")
