from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sktime.transformations.series.vmd import VmdTransformer

if __name__ == "__main__":
    print("+" * 8, " `vmd_features.py` started. ", "+" * 8)

    # Set paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs/models/vmd_hyperparams_config.yaml"
    data_path = BASE_DIR / "data/processed"
    model_path = BASE_DIR / "models"

    # Set configs
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # For each train set, test set, and model save folder --> loop
    for train, test, folder in zip(
        ["train", "train_val"], ["val", "test"], ["partial", "full"]
    ):
        ## Load data
        df_train = pd.read_parquet(data_path / f"ann/{train}.parquet")
        df_test = pd.read_parquet(data_path / f"ann/{test}.parquet")

        ## Get price arrays
        train_price = df_train["price"].to_numpy()
        test_price = df_test["price"].to_numpy()

        ## Instantiate VMD
        vmd = VmdTransformer(**cfg["vmd"])
        vmd.set_random_state(cfg["random_state"])

        ## Fit / transform price arrays
        train_imfs = vmd.fit_transform(train_price)
        test_imfs = vmd.transform(test_price)

        ## Add IMFs to DataFrames
        df_train[[f"imf{i}" for i in range(1, 6)]] = train_imfs
        df_test[[f"imf{i}" for i in range(1, 6)]] = test_imfs

        ## Add residual IMF (raw - sum(IMFs)) to DataFrames
        df_train["imf_resid"] = train_price - np.sum(train_imfs, axis=1)  # type: ignore
        df_test["imf_resid"] = test_price - np.sum(test_imfs, axis=1)  # type: ignore

        ## Save IMF DataFrame
        df_train.to_parquet(data_path / f"hybrid/{train}.parquet", index=True)
        df_test.to_parquet(data_path / f"hybrid/{test}.parquet", index=True)

        ## Save VMD model
        dump(vmd, BASE_DIR / f"models/hybrid/{folder}/vmd.pkl")

    print("+" * 8, " `vmd_features.py` completed. ", "+" * 8)
