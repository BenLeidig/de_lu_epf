import numpy as np
import pandas as pd
from sktime.transformations.series.vmd import VmdTransformer


def vmd_fit_transform(K: int, alpha: int, random_state: int, y: np.ndarray):
    """Fit and transform the provided array using VMD with the provided hyperparameters.

    Args:
        K (int): Number of modes for VMD.
        alpha (int): Penalty term for VMD.
        random_state (int): Random state.
        y (np.ndarray): numpy array to (fit and) transform.

    Returns:
        np.ndarray: (len(y), K) matrix of VMD transformed y values.
    """
    vmd = VmdTransformer(K=K, alpha=alpha)
    vmd.set_random_state(random_state)
    return vmd.fit_transform(y)


def fc(x: np.ndarray):
    """Calculate the center frequency (f_c) of the provided array.

    Args:
        x (np.ndarray): Array to calculate f_c for.

    Returns:
        np.float64: Center frequency of x.
    """
    X = np.fft.rfft(x)
    power = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(len(x))
    return float(np.sum(freqs * power) / np.sum(power))


def abs_corrcoef(x: np.ndarray, y: np.ndarray):
    """Calculate the absolute Pearson correlation coefficient between two arrays.

    Args:
        x (np.ndarray): numpy array.
        y (np.ndarray): numpy array.

    Returns:
        np.float64: Absolute value of the Pearson correlation coefficient.
    """
    m = np.corrcoef(x, y)
    corrcoef = m[0, 1]
    return float(np.abs(corrcoef))


def vmd_hpo(K: int, alpha: int, random_state: int, y: np.ndarray):
    """Finds the center frequencies and correlation (Pearson, with the raw signal) for

    Args:
        K (int): Number of modes for VMD.
        alpha (int): Penalty term for VMD.
        random_state (int): Random state.
        y (np.ndarray): numpy array to (fit and) transform.

    Returns:
        tuple[dict[int, float], dict[int, float]]: Tuple of (center frequency dictionary, and absolute Pearson correlation coefficient dictionary)
    """
    imfs = vmd_fit_transform(K=K, alpha=alpha, random_state=random_state, y=y)
    fc_dict = {i: fc(imfs[:, i]) for i in range(imfs.shape[1])}  # type: ignore
    corr_dict = {i: abs_corrcoef(imfs[:, i], y) for i in range(imfs.shape[1])}  # type: ignore
    return fc_dict, corr_dict


def find_fc_max(df: pd.DataFrame):
    """Find the max center frequency for all modes for each K.

    Args:
        df (pd.DataFrame): DataFrame containing K, fc as columns.

    Returns:
        pd.DataFrame: DataFrame with the max center frequency for all modes for each K.
    """
    df_out = df[["K", "fc"]].groupby("K", as_index=False).agg("max")
    df_out = df_out.rename(columns={"fc": "fc_max"})  # type: ignore
    return df_out


def find_cfr_min_max(df: pd.DataFrame):
    """Find the center frequency ratio (CFR) using the min-max method.

    Args:
        df (pd.DataFrame): DataFrame containing K, fc as columns.

    Returns:
        pd.DataFrame: DataFrame with the CFR for each K.
    """
    df_out = df[["K", "fc"]].groupby("K").agg("max") / df[["K", "fc"]].groupby("K").agg(
        "min"
    )
    df_out = df_out.reset_index().rename(columns={"fc": "CFR (max/min)"})
    return df_out


def find_cfr_roc(df: pd.DataFrame):
    """Find the center frequency ratio (CFR) using the rate-of-change method.

    Args:
        df (pd.DataFrame): pd.DataFrame containing K, fc as columns.

    Returns:
        pd.DataFrame: DataFrame with the CFR for each K.
    """
    df_out = df[["K", "fc"]].groupby("K").agg("max") / df[["K", "fc"]].groupby("K").agg(
        "min"
    )
    df_out = df_out.reset_index().rename(columns={"fc": "CFR (max/min)"})
    df_out["CFR (RoC)"] = df_out["CFR (max/min)"] / df_out["CFR (max/min)"].shift(1)
    df_out = df_out[["K", "CFR (RoC)"]]
    return df_out


def find_cfr_K(df: pd.DataFrame):
    """Find the center frequency ratio (CFR) using the recursive method.

    Args:
        df (pd.DataFrame): pd.DataFrame containing K, fc as columns.

    Returns:
        pd.DataFrame: DataFrame with the CFR for each K.
    """
    df_out = (
        df[["K", "fc"]]
        .groupby("K", as_index=False)
        .agg("max")
        .sort_values("K", ascending=True)
    )  # type: ignore
    df_out["CFR (K/(K-1))"] = df_out["fc"] / df_out["fc"].shift(1)
    df_out = df_out[["K", "CFR (K/(K-1))"]]
    return df_out


def find_fc_sd(df: pd.DataFrame):
    """Find the standard deviation of center frequencies for each K.

    Args:
        df (pd.DataFrame): pd.DataFrame containing K, fc as columns.

    Returns:
        pd.DataFrame: DataFrame with the standard deviation for each K.
    """
    df_out = df[["K", "fc"]].groupby("K", as_index=False).agg("std")
    df_out = df_out.rename(columns={"fc": "fc_sd"})  # type: ignore
    return df_out
