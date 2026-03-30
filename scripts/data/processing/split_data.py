from pathlib import Path

import pandas as pd
import yaml

from data.processing import save_splits

if __name__ == "__main__":
    print("+" * 8, " `split_data.py` started. ", "+" * 8)

    # Set paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs"
    data_path = BASE_DIR / "data"

    # Set configs
    with open(cfg_path / "data/process_config.yaml") as f:
        cfg = yaml.safe_load(f)["dt_range"]

    # Load data
    df = pd.read_parquet(data_path / "processed/processed.parquet")
    df_dmf = pd.read_parquet(data_path / "processed/dmf_data/processed.parquet")

    # Get split list (list of dates for each split)
    split_names = ["train", "val", "test"]
    split_list = []
    for k in split_names:
        split_list.append((cfg[k]["start"], cfg[k]["end"]))

    #### #### ANN-specific data splits #### ####
    save_splits(df, split_list, path=data_path / "processed/ann_data")
    print("-" * 8, "Saved ANN splits.", "-" * 8)

    #### #### DMF-specific data splits #### ####
    save_splits(df_dmf, split_list, path=data_path / "processed/dmf_data")
    print("-" * 8, "Saved DMF splits.", "-" * 8)

    print("+" * 8, " `split_data.py` completed. ", "+" * 8)
