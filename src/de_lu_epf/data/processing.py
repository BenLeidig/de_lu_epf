from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


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


def save_splits(df: pd.DataFrame, split_list: list, path: Path):
    """Save the train, val, train_val, test splits for the given DataFrame, specified list of date tuples, and save path.

    Args:
        df (pd.DataFrame): DataFrame to split. Must have a datetime (UTC) index.
        split_list (list): List of tuples with (start, end) datetime (UTC) format for each split.
        path (Path): Path to save splits.
    """
    split_names = ["train", "val", "test"]
    splits = get_splits(df, split_list)

    for n, s in zip(split_names, splits):
        s.to_parquet(path / f"{n}.parquet", index=True)

    train_val = pd.concat(splits[:2], axis=0)
    train_val.to_parquet(path / "train_val.parquet", index=True)


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
        scaler.fit_transform(df_train), columns=df_train.columns, index=df_train.index
    )
    df_test_scaled = pd.DataFrame(
        scaler.transform(df_test), columns=df_test.columns, index=df_test.index
    )
    return scaler, df_train_scaled, df_test_scaled
