from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_dt_col(j: dict, dt_col: str, dt_params: dict, dt_delta: int, empty: bool):
    """Creates a DataFrame with a datetime column the provided datatime data.

    Args:
        j (dict): Dictionary data.
        dt_col (str): Key for the values containing datetime information.
        dt_params (dict): Parameters for the pd.DatetimeIndex conversion.
        dt_delta (int): Time offset in hours (i.e. alignment is necessary).
        empty (bool): Whether or not to return an empty DataFrame (besides for the datetime column).

    Returns:
        pd.DataFrame: DataFrame with a singular column of datetime data from the converted datetime column.
    """
    df = pd.DataFrame() if empty else pd.DataFrame(j)
    df["datetime"] = pd.to_datetime(
        j[dt_col] if empty else df[dt_col], **dt_params
    ) + pd.Timedelta(hours=dt_delta)
    return df


def normalize_external(j: dict, df: pd.DataFrame, nest_key: str):
    """Normalize the nested dictionary structure and return it as a DataFrame.

    Args:
        j (dict): Dictionary data.
        df (pd.DataFrame): Empty DataFrame.
        nest_key (str): Specified key for the 'nests' of the nested structure.

    Returns:
        pd.DataFrame: Dataframe of normalized structure.
    """
    for col_data in j[nest_key]:
        col = col_data["name"].lower().replace(" ", "_").replace("-", "_")
        df[col] = col_data["data"]
    return df


def make_hr_freq(df: pd.DataFrame):
    """Select only hourly interval data of the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to modify (not in-place).

    Returns:
        pd.DataFrame: Modified DataFrame of hourly intervals.
    """
    return df[df["datetime"].dt.minute == 0]


def reframe_df(df: pd.DataFrame, start: str, end: str):
    """Reframe the specified DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to be reframed (not in-place).
        start (str): (Inclusive) Datetime-like string for the first datetime.
        end (str): (Inclusive) Datetime-lik string for the last datetime.

    Returns:
        pd.DataFrame: Reframed DataFrame
    """
    return df[(df["datetime"] >= start) & (df["datetime"] <= end)]


def subset_df(df: pd.DataFrame, subset: list):
    """Return a subset of the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to subset.
        subset (list): List of columns for the subset.

    Returns:
        pd.DataFrame: Subsetted DataFrame.
    """
    return df[subset]


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
