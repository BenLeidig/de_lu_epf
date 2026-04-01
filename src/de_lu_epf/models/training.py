from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from yaml import safe_load

from de_lu_epf.models.architectures import DirectMultiStepForecaster
from de_lu_epf.models.hpo.dmf_tuning import supports_parallel, supports_random_state


def get_fitted_dmf(model_name: str):

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    CONFIG_DIR = BASE_DIR / "configs/models"
    DATA_DIR = BASE_DIR / "data/processed/dmf"

    with open(CONFIG_DIR / "preprocess_config.yaml") as f:
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
        with open(CONFIG_DIR / "dmf_hyperparams_config.yaml") as f:
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
