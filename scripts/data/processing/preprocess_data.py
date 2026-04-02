from pathlib import Path

import pandas as pd
import yaml
from joblib import dump

from de_lu_epf.data.processing import get_scaled

if __name__ == "__main__":
    print("+" * 8, " `scale.py` started. ", "+" * 8)

    # Set paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs"
    data_path = BASE_DIR / "data/processed"

    # Set configs
    with open(cfg_path / "models/preprocess_config.yaml") as f:
        preprocess_cfg = yaml.safe_load(f)

    # For each dataset --> loop
    for set in ["ann", "dmf", "hybrid"]:
        ## For each train set, test set, and partial / full folder --> loop
        for train, test, folder in zip(
            ["train", "train_val"], ["val", "test"], ["partial", "full"]
        ):
            ### Load data
            df_train = pd.read_parquet(data_path / f"{set}_data/{train}.parquet")
            df_test = pd.read_parquet(data_path / f"{set}_data/{test}.parquet")

            if set == "dmf":  # Don't scale responses for DMF - only scale features
                dmf_features = preprocess_cfg["dmf"]["features"]
                dmf_responses = preprocess_cfg["dmf"]["targets"]
                X_train = df_train.loc[:, dmf_features]
                y_train = df_train[dmf_responses]
                X_test = df_test.loc[:, dmf_features]
                y_test = df_test[dmf_responses]

                ### Scaled features
                scaler, X_train_scaled, X_test_scaled = get_scaled(X_train, X_test)

                ### Concatenate scaled features and unscaled responses
                df_train_scaled = pd.concat([X_train_scaled, y_train], axis=1)
                df_test_scaled = pd.concat([X_test_scaled, y_test], axis=1)

            else:
                ### Scale data
                scaler, df_train_scaled, df_test_scaled = get_scaled(df_train, df_test)

            ### Save DataFrames
            df_train_scaled.to_parquet(
                data_path / f"{set}_data/{train}_scaled.parquet", index=True
            )
            df_test_scaled.to_parquet(
                data_path / f"{set}_data/{test}_scaled.parquet", index=True
            )

            ### Save scaler
            dump(scaler, BASE_DIR / f"models/{set}_models/{folder}/scaler.pkl")

    print("+" * 8, " `scale.py` completed. ", "+" * 8)
