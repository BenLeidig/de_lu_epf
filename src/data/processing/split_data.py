from pathlib import Path

import pandas as pd
import yaml


def get_splits(df: pd.DataFrame, split_list: list):
    """Get the splits for the provided DataFrame at the specified split datetimes.

    Args:
        df (pd.DataFrame): DataFrame to split. Must have a datetime (UTC) index.
        split_list (list): List of tuples with (start, end) datetime (UTC) format for each split.

    Returns:
        tuple: Tuple of DataFrame splits.
    """
    splits = []
    for start, end in split_list:
        idx = df.index
        start = idx.searchsorted(pd.to_datetime(start, utc=True))
        end = idx.searchsorted(pd.to_datetime(end, utc=True))
        splits.append(df.iloc[start:end])
    return tuple(splits)


def save_splits(df: pd.DataFrame, split_list: list, path):
    """Save the train, val, train_val, test splits for the given DataFrame, specified list of date tuples, and save path.

    Args:
        df (pd.DataFrame): DataFrame to split. Must have a datetime (UTC) index.
        split_list (list): List of tuples with (start, end) datetime (UTC) format for each split.
        path (_type_): Path to save splits.
    """
    split_names = ["train", "val", "test"]
    splits = get_splits(df, split_list)

    for n, s in zip(split_names, splits):
        s.to_parquet(path / f"{n}.parquet", index=True)

    train_val = pd.concat(splits[:2], axis=0)
    train_val.to_parquet(path / "train_val.parquet", index=True)


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

    # Get split list (list of dates for each split)
    split_names = ["train", "val", "test"]
    split_list = []
    for k in split_names:
        split_list.append((cfg[k]["start"], cfg[k]["end"]))

    #### #### ANN-specific data splits #### ####
    save_splits(df, split_list, path=data_path / "processed/ann_data")
    print("Saved ANN splits.")

    #### #### DMF-specific data splits #### ####
    # coming soon...

    print("+" * 8, " `split_data.py` completed. ", "+" * 8)
