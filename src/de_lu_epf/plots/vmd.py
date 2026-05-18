import pandas as pd

from de_lu_epf.plots.utils import plot_stacked_series


def plot_vmd_decomposition(
    index,
    raw_series_dict: dict,
    imf_series_dict: dict,
    suptitle: str = None,  # type: ignore
):
    series_dict = raw_series_dict.copy()

    for imf, values in imf_series_dict.items():
        label = "Residual" if "resid" in imf.lower() else imf.upper()
        series_dict[label] = values

    return plot_stacked_series(index=index, series_dict=series_dict, suptitle=suptitle)