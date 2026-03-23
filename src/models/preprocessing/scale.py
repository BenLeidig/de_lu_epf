from pathlib import Path

import pandas as pd
import yaml
from joblib import dump
from sklearn.preprocessing import StandardScaler


def get_scaled(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Scale the provided train and test datasets.

    Args:
        df_train (pd.DataFrame): Train data to fit and transform.
        df_test (pd.DataFrame): Test data to transform.

    Returns:
        tuple: Tuple of (fitted scaler, scaled train data, scaled test data).
    """
    scaler = StandardScaler()
    df_train_scaled = pd.DataFrame(
        scaler.fit_transform(df_train), columns=df_train.columns
    )
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)
    return scaler, df_train_scaled, df_test_scaled


if __name__ == "__main__":
    print("+" * 8, " `scale.py` started. ", "+" * 8)

    # Set paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs/model/hyperparams_config.yaml"
    data_path = BASE_DIR / "data/processed"

    # Set configs
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # For each dataset --> loop
    for set in ["ann", "vmd"]:  #### !!!! ADD dmf_data AT LATER POINT !!!! ####
        ## For each train set, test set, and partial / full folder --> loop
        for train, test, folder in zip(
            ["train", "train_val"], ["val", "test"], ["partial", "full"]
        ):
            ### Load data
            df_train = pd.read_parquet(data_path / f"{set}_data/{train}.parquet")
            df_test = pd.read_parquet(data_path / f"{set}_data/{test}.parquet")

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
