from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.vmd import VmdTransformer

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    cfg_path = BASE_DIR / "configs/model/hyperparams_config.yaml"
    interim_data_path = BASE_DIR / "data/interim"
    processed_data_path = BASE_DIR / "data/processed"
    model_path = BASE_DIR / "models"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    vmd_cfg = cfg["vmd"]

    for train, test, mod_dir in zip(
        ["train", "train_val"], ["val", "test"], ["partial", "full"]
    ):
        df_train = pd.read_parquet(interim_data_path / f"ann_{train}.parquet")
        df_test = pd.read_parquet(interim_data_path / f"ann_{test}.parquet")

        train_price = df_train["price"].to_numpy()
        test_price = df_test["price"].to_numpy()

        vmd = VmdTransformer(**vmd_cfg)
        vmd.set_random_state(cfg["random_state"])

        train_imfs = vmd.fit_transform(train_price)
        test_imfs = vmd.transform(test_price)

        df_train[[f"imf{i}" for i in range(1, 6)]] = train_imfs
        df_test[[f"imf{i}" for i in range(1, 6)]] = test_imfs

        df_train["imf_resid"] = train_price - np.sum(train_imfs)
        df_test["imf_resid"] = test_price - np.sum(test_imfs)

        scaler = StandardScaler()
        df_train_scaled = pd.DataFrame(
            scaler.fit_transform(df_train), columns=df_train.columns
        )
        df_test_scaled = pd.DataFrame(
            scaler.transform(df_test), columns=df_test.columns
        )

        df_train.to_parquet(processed_data_path / f"vmd_{train}.parquet", index=True)
        df_test.to_parquet(processed_data_path / f"vmd_{test}.parquet", index=True)
        df_train_scaled.to_parquet(
            processed_data_path / f"vmd_{train}_scaled.parquet", index=True
        )
        df_test_scaled.to_parquet(
            processed_data_path / f"vmd_{test}_scaled.parquet", index=True
        )

        dump(vmd, model_path / f"{mod_dir}/vmd.pkl")
        dump(scaler, model_path / f"{mod_dir}/scaler.pkl")
