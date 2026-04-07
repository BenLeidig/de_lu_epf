import json
from functools import reduce
from pathlib import Path

import pandas as pd
import yaml

from de_lu_epf.data.processing import (
    add_dt_col,
    make_hr_freq,
    normalize_external,
    subset_df,
)

if __name__ == "__main__":
    print("+" * 8, " `merge_data.py` started. ", "+" * 8)

    # Set paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs"
    external_data_path = BASE_DIR / "data/external"
    interim_data_path = BASE_DIR / "data/interim"

    # Set configs
    with open(cfg_path / "data/process_config.yaml") as f:
        cfg = yaml.safe_load(f)
    source_cfg = cfg["data_source"]

    merge_list = []  ## Listing of separate data source DataFrames

    # Create a DataFrame with the average German weather data (across 4 main cities)
    concat_list = []
    for f_path in external_data_path.glob("*weather.json"):
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
    with open(external_data_path / "price.json") as f:
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
    with open(external_data_path / "production.json") as f:
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
    with open(external_data_path / "trade.json") as f:
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

    # Saving data
    df.to_parquet(interim_data_path / "merged.parquet", index=True)

    print("+" * 8, " `merge_data.py` completed. ", "+" * 8)
