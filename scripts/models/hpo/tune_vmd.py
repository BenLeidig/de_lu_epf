from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from de_lu_epf.models.hpo.vmd_tuning import (
    find_cfr_K,
    find_cfr_min_max,
    find_cfr_roc,
    find_fc_max,
    find_fc_sd,
    vmd_hpo,
)

if __name__ == "__main__":
    print("+" * 8, " `tune_vmd.py` started. ", "+" * 8)

    # Set paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    cfg_path = BASE_DIR / "configs/models/hpo_config.yaml"
    data_path = BASE_DIR / "data/processed/processed.parquet"
    output_path = BASE_DIR / "reports/figures/vmd_hpo"

    # Set configs
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["vmd"]

    # Read data
    train_val_price = pd.read_parquet(data_path)["price"].to_numpy()

    fc_dict = {}
    corr_dict = {}
    records = []
    K_min = cfg["K"]["min"]
    K_max = cfg["K"]["max"]
    alpha = cfg["alpha"]
    random_state = cfg["random_state"]

    # Pack IMF analysis data into DataFrame
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

    # Iterative plotting of center frequency figures
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

    # Plot IMF correlation analysis
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

    print("+" * 8, " `tune_vmd.py` completed. ", "+" * 8)
