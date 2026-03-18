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


if __name__ == "__main__":
    # Config
    with open("configs/data/features_config.yaml") as f:
        cfg = yaml.safe_load(f)
    dt_cfg = cfg["dt_range"]
    rolling_mean_lengths = cfg["rolling_stats"]["mean"]
    rolling_var_lengths = cfg["rolling_stats"]["var"]

    # Data reading
    df = pd.read_parquet("data/interim/processed.parquet")

    # Calculating rolling mean and variance
    for r in rolling_mean_lengths:
        df[f"{r}_rmean_price"] = df["price"].rolling(window=r).mean()
    for r in rolling_var_lengths:
        df[f"{r}_rvar_price"] = df["price"].rolling(window=r).var()

    # Train-val-test splits
    split_list = []
    for k in ["train", "val", "test"]:
        split_list.append((dt_cfg[k]["start"], dt_cfg[k]["end"]))

    splits = get_splits(df, split_list)
    split_names = ["train", "val", "test"]
    for n, s in zip(split_names, splits):
        s.to_parquet(f"data/interim/{n}.parquet", index=True)

    train_val = pd.concat(splits[:2], axis=0)
    train_val.to_parquet("data/interim/train_val.parquet", index=True)
