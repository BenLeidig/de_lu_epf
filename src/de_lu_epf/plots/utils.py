from pathlib import Path

import matplotlib.pyplot as plt
from yaml import safe_load


def update_matplotlib_params():
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    CFG_DIR = BASE_DIR / "configs"
    with open(CFG_DIR / "plots_configs.yaml") as f:
        cfg = safe_load(f)["matplotlib"]
    plt.rcParams.update(cfg)


def plot_series(index, values, ax, color, label, linewidth=1.0, axhline: bool = True):
    if axhline:
        ax.axhline(0, linewidth=0.7, color="black")
    ax.plot(index, values, linewidth=linewidth, c=color, label=label, alpha=0.8)
    ax.grid(axis="x", linestyle="--")


def plot_stacked_series(index, series_dict: dict, suptitle: str = None):  # type: ignore
    fig, axes = plt.subplots(len(series_dict), sharex=True, figsize=(6, 4))
    colors = plt.colormaps["tab10"].colors  # type: ignore

    for i, (label, values) in enumerate(series_dict.items()):
        ax = axes[i]
        color = colors[i % len(colors)]
        plot_series(index=index, values=values, ax=ax, color=color, label=label)
        ax.set_ylabel(label)

    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()

    return fig, axes
