import json
from functools import reduce
from pathlib import Path
from re import split

import pandas as pd
import yaml


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


# Set configs
with open("configs/data/process_config.yaml") as f:
    cfg = yaml.safe_load(f)
source_cfg = cfg["data_source"]
dt_cfg = cfg["dt_range"]

dir = Path("data/external")  ## External data directory
merge_list = []  ## Listing of separate data source DataFrames


# Create a DataFrame with the average German weather data (across 4 main cities)
concat_list = []
for f_path in dir.glob("*weather.json"):
    t_cfg = source_cfg["weather"]
    with open(f_path) as f:
        j = json.load(f)
        t_df = add_dt_col(
            j["weather"],
            dt_col=t_cfg["dt_col"],
            dt_params=t_cfg["dt_params"],
            dt_delta=t_cfg["dt_delta"],
            empty=False,
        )
        t_df = subset_df(t_df, t_cfg["variables"] + ["datetime"])
        concat_list.append(t_df)
weather_df = pd.concat(concat_list, axis=0)
weather_df = weather_df.groupby("datetime", as_index=False).agg("mean")
merge_list.append(weather_df)

# Create a DataFrame with day-ahead price data
price_cfg = source_cfg["price"]
with open("data/external/price.json") as f:
    j = json.load(f)
    price_df = add_dt_col(
        j,
        dt_col=price_cfg["dt_col"],
        dt_params=price_cfg["dt_params"],
        dt_delta=price_cfg["dt_delta"],
        empty=False,
    )
    price_df = subset_df(price_df, price_cfg["variables"] + ["datetime"])
    merge_list.append(price_df)

# Create a DataFrame with electricity production data
production_cfg = source_cfg["production"]
with open("data/external/production.json") as f:
    j = json.load(f)
    production_df = add_dt_col(
        j,
        dt_col=production_cfg["dt_col"],
        dt_params=production_cfg["dt_params"],
        dt_delta=production_cfg["dt_delta"],
        empty=True,
    )
    production_df = normalize_external(
        j, production_df, nest_key=production_cfg["nest_key"]
    )
    production_df = make_hr_freq(production_df)
    production_df = subset_df(production_df, production_cfg["variables"] + ["datetime"])  # type: ignore
    merge_list.append(production_df)

# Create a DataFrame with electricity trading data
trade_cfg = source_cfg["trade"]
with open("data/external/trade.json") as f:
    j = json.load(f)
    trade_df = add_dt_col(
        j,
        dt_col=trade_cfg["dt_col"],
        dt_params=trade_cfg["dt_params"],
        dt_delta=trade_cfg["dt_delta"],
        empty=True,
    )
    trade_df = normalize_external(j, trade_df, nest_key=trade_cfg["nest_key"])
    trade_df = make_hr_freq(trade_df)
    trade_df = trade_df.rename({"sum": "sum_cbet"}, axis=1)
    trade_df = subset_df(trade_df, trade_cfg["variables"] + ["datetime"])  # type: ignore
    merge_list.append(trade_df)

# Merge DataFrames
df = reduce(
    lambda l, r: l.merge(r, on="datetime", how="outer"), merge_list
).sort_values(by="datetime", ascending=True)
df = df.set_index("datetime")

# Calculating rolling mean and variance
df["168_mean_price"] = df["price"].rolling(window=24 * 7 * 4).mean()
df["24_mean_price"] = df["price"].rolling(window=24).mean()

df["168_var_price"] = df["price"].rolling(window=24 * 7 * 4).var()
df["24_var_price"] = df["price"].rolling(window=24).var()

#### ANN-Specific Data Adjusting ####
ann_df = df.copy(deep=True)

# Lagging non-price covariates to abide by day-head constraint
## i.e., prices are determined at noon the day before, so we have price data up until last
## hour of the day (since prices are pre-determined), but not exogenous (non-price) data.
for col in ann_df.columns:
    if "price" not in col:
        ann_df.loc[:, col] = ann_df[col].shift(12)

# Train-val-test splits
split_list = []
for k in ["train", "val", "test"]:
    split_list.append((dt_cfg[k]["start"], dt_cfg[k]["end"]))

ann_splits = get_splits(df, split_list)
ann_split_names = ["ann_train", "ann_val", "ann_test"]
for n, s in zip(ann_split_names, ann_splits):
    s.to_parquet(f"data/interim/{n}.parquet")

ann_train_val = pd.concat(ann_splits[:2], axis=0)
ann_train_val.to_parquet("data/interim/ann_train_val.parquet")
