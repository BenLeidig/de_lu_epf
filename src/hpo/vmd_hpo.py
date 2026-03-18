from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
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
    fc_dict = {i: fc(imfs[:, i]) for i in range(imfs.shape[1])}
    corr_dict = {i: abs_corrcoef(imfs[:, i], y) for i in range(imfs.shape[1])}
    return fc_dict, corr_dict


def find_fc_max(df: pd.DataFrame):
    """Find the max center frequency for all modes for each K.

    Args:
        df (pd.DataFrame): DataFrame containing K, fc as columns.

    Returns:
        pd.DataFrame: DataFrame with the max center frequency for all modes for each K.
    """
    df_out = df[["K", "fc"]].groupby("K", as_index=False).agg("max")
    df_out = df_out.rename(columns={"fc": "fc_max"})
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
    )
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
    df_out = df_out.rename(columns={"fc": "fc_sd"})
    return df_out


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    cfg_path = BASE_DIR / "configs/model/hpo_config.yaml"
    data_path = BASE_DIR / "data/interim/train_val.parquet"
    output_path = BASE_DIR / "reports/figures/vmd_hpo"

    with open("configs/model/hpo_config.yaml") as f:
        cfg = yaml.safe_load(f)["vmd"]

    train_val_price = pd.read_parquet(data_path)["price"].to_numpy()

    fc_dict = {}
    corr_dict = {}
    records = []
    K_min = cfg["K"]["min"]
    K_max = cfg["K"]["max"]
    alpha = cfg["alpha"]
    random_state = cfg["random_state"]

    for K in range(K_min, K_max + 1):
        fc_dict[K], corr_dict[K] = vmd_hpo(
            K=K, alpha=alpha, random_state=random_state, y=train_val_price
        )

        for imf_idx, fc_val in fc_dict[K].items():
            records.append(
                {
                    "K": K,
                    "IMF": imf_idx + 1,
                    "fc": fc_val,
                    "corr": corr_dict[K][imf_idx],
                }
            )
        print(f"{K} complete.")

    df = pd.DataFrame(records)
    df_fc = reduce(
        lambda l, r: l.merge(r, on="K", how="outer"),
        [
            find_fc_max(df),
            find_cfr_min_max(df),
            find_cfr_roc(df),
            find_cfr_K(df),
            find_fc_sd(df),
        ],
    ).sort_values(by="K", ascending=True)

    cols = ["fc_max", "CFR (max/min)", "CFR (RoC)", "CFR (K/(K-1))", "fc_sd"]
    labels = [
        r"$\max F_c$",
        r"$\frac{\max F_c}{\min F_c}$",
        r"CFR RoC",
        r"$\frac{K}{K-1}$ CFR by $K$",
        r"$\sigma^2_{\mathbf{F_c}}$",
    ]
    titles = [
        r"Max Center Frequency by $K$",
        r"Center Frequency Ratio (CFR) ($\frac{\max F_c}{\min F_c}$) by $K$",
        r"Center Frequency Ratio (CFR) Rate of Change (RoC) by $K$",
        r"Center Frequency Ratio (CFR) ($\frac{K}{K-1}$) by $K$",
        r"Center Frequency Standard Deviation by $K$",
    ]
    for col, lab, title in zip(cols, labels, titles):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df_fc["K"], df_fc[col], c="tab:blue", linewidth=2, marker="o")
        ax.set_ylabel(lab)
        ax.set_xlabel(r"Number of Modes ($K$)")
        ax.set_xticks([_ for _ in range(2, 21)])
        ax.grid(linewidth=0.5, linestyle="--")
        ax.axvline(5, color="tab:red", linestyle="--")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(
            output_path / f"{col.lower().replace(' ', '_').replace('/', '_')}.svg"
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(
        data=df,
        x="IMF",
        y="corr",
        hue="K",
        palette="tab10",
        alpha=0.7,
        linewidth=2,
        legend=False,
    )
    ax.set_ylabel(r"$|\rho_{\text{IMF}_{i,K},\text{price}}|$")
    ax.set_xlabel("Intrinsic Mode Function (IMF)")
    ax.set_title("Correlation with between IMF and Raw Signal by K")
    ax.set_xticks([_ for _ in range(1, 21)])
    ax.grid(linewidth=0.5, linestyle="--")
    ax.axvline(5, color="tab:red", linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path / "imf_correlation.svg")
