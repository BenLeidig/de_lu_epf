from pathlib import Path

import pandas as pd
import yaml

if __name__ == "__main__":
    print("+" * 8, " `build_features.py` started. ", "+" * 8)

    # Set paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    config_path = BASE_DIR / "configs"
    data_path = BASE_DIR / "data"

    # Set configs
    with open(config_path / "data/features_config.yaml") as f:
        cfg = yaml.safe_load(f)
    rolling_mean_lengths = cfg["rolling_stats"]["mean"]
    rolling_var_lengths = cfg["rolling_stats"]["var"]

    # Load data
    df = pd.read_parquet(data_path / "interim/aligned.parquet")

    # Calculating rolling mean and variance
    for r in rolling_mean_lengths:
        df[f"{r}_rmean_price"] = df["price"].rolling(window=r).mean()
    for r in rolling_var_lengths:
        df[f"{r}_rvar_price"] = df["price"].rolling(window=r).var()

    # Save data
    df.to_parquet(data_path / "processed/processed.parquet", index=True)

    print("+" * 8, " `build_features.py` completed. ", "+" * 8)
