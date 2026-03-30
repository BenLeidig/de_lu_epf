from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    print("+" * 8, " `align_data.py` started. ", "+" * 8)

    # Set paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    data_path = BASE_DIR / "data"

    # Loading data
    df = pd.read_parquet(data_path / "interim/merged.parquet")

    # Lagging non-price covariates to abide by day-head constraint
    ## i.e., prices are determined at noon the day before, so we have price data up until last
    ## hour of the day (since prices are pre-determined), but not exogenous (non-price) data.
    for col in df.columns:
        if "price" not in col:
            df.loc[:, col] = df[col].shift(12)

    # Saving data
    df.to_parquet(data_path / "interim/aligned.parquet", index=True)

    print("+" * 8, " `align_data.py` completed. ", "+" * 8)
