from pathlib import Path
from typing import Union

import pandas as pd


def create_dmf_cols(
    df: Union[pd.DataFrame, pd.Series],
    constructor_dict: dict,
    var: str,
    times: range,
    lags: range,
    day: str,
):
    """Specialized function for creating aligned data for direct multi-step forecasting (DMF) models in this repository.
    Args:
        df (pd.DataFrame): DataFrame to fetch pd.Series column data from.
        constructor_dict (dict): Constructor dictionary in use for building aligned DMF DataFrame.
        var (str): Name of column to align.
        times (range): Range representing the realtime values with respect to `day` specification.
        lags (range): Range of lags that correspond with `times` range with respect to `day` specification.
        day (str): Whether the alignment is with respect to the previous day ("prev") or the forecasting day ("forecast").

    Raises:
        ValueError: If `day` is not one of ("prev", "forecast")

    Returns:
        dict: Modified constructor dictionary, constructor_dict.
    """
    for time, lag in zip(times, lags):
        # Column naming logic (see comment below).
        if day == "prev":
            key = f"{time}_{var}"
        elif day == "forecast":
            key = f"{var}_{time}"
        else:
            raise ValueError("Paremeter `day` must be one of ('prev', 'forecast').")

        # Save aligned column data to list
        val = df[var].shift(lag).to_list()

        # Assign list to key name
        constructor_dict[key] = val

    # Return the modified constructor dictionary
    return constructor_dict


if __name__ == "__main__":
    print("+" * 8, " `dmf_features.py` started. ", "+" * 8)

    # Set paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs/data/process_config.yaml"
    data_path = BASE_DIR / "data/processed"

    df = pd.read_parquet(data_path / "processed.parquet")
    df = df[df.index.hour == 0]  # type: ignore

    # NOTE: Naming convention:
    ## `**_{variable}` signifies the `**`th hour of the current day.
    ## `{variable}_**` signifies the `**`th hour of the next day (the forecasted day).

    dict_dmf = {}
    dict_dmf["datetime"] = df.index.to_list()

    # Adding the time aligned features
    ## Logic: we get a column for each feature for all past hours of the past day
    ## (i.e., 00:00:00 - 00:12:00 for exogenous variables and 00:00:00 - 00:23:00 for
    ## price related features since we have yesterday's set values for the future hours
    ## of the past day.)
    ## NOTE: the data is 'centered' on 00:00:00, the first hour of the forecasting day.
    ### Thus, the "past day" is really the current day, and the "forecasting day" is
    ### tomorrow.
    for var in df.columns:
        if "price" not in var:
            dict_dmf = create_dmf_cols(
                df, dict_dmf, var, range(0, 13), range(24, 11, -1), "prev"
            )
        else:
            dict_dmf = create_dmf_cols(
                df, dict_dmf, var, range(0, 24), range(24, 0, -1), "prev"
            )

    # Adding the multivariate responses
    ## Logic: new response (i.e. `price`) for each hour of the next day
    dict_dmf = create_dmf_cols(
        df, dict_dmf, "price", range(0, 24), range(0, -24, -1), "forecast"
    )

    # Convert constructor dictionary to DataFrame
    df_dmf = pd.DataFrame.from_dict(dict_dmf, orient="columns")
    df_dmf = df_dmf.set_index("datetime")

    # Save DataFrame
    df_dmf.to_parquet(data_path / "dmf_data/processed.parquet", index=True)

    print("+" * 8, " `dmf_features.py` completed. ", "+" * 8)
