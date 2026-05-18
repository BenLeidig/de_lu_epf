from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ptitprince as pt
from yaml import safe_load

## Some utility code for scripted / notebook generation of visualizations.
## Fairly straightfoward, so I don't believe many code comments are
## necessary. It should be noted that some of these functions are
## experimental and may not be used in the final repository code.


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


def raincloudplot(x, y, palette, data, ax):
    pt.half_violinplot(
        x=x,
        y=y,
        scale="area",
        palette=palette,
        inner=None,
        data=data,
        width=1,
        ax=ax,
        orient="h",
    )

    groups = data[y].unique()
    for i, g in enumerate(groups):
        group_data = data[data[y] == g]
        y_jitter = i + np.random.uniform(high=0.2, size=len(group_data))
        x_jitter = group_data[x]
        alpha = np.clip(2 / np.sqrt(len(group_data)), 0.05, 0.6)
        ax.scatter(x_jitter, y_jitter, color=palette[i], alpha=alpha)

    shift = 0.1
    positions = [i + shift for i in range(len(groups))]
    boxplot_data = [data[data[y] == g][x].values for g in groups]

    medianprops = {"linewidth": 1.5, "color": "black", "solid_capstyle": "butt"}
    boxprops = {"linewidth": 1.5, "color": "darkgray"}

    ax.boxplot(
        boxplot_data,
        vert=False,
        positions=positions,
        manage_ticks=False,
        showfliers=False,
        showcaps=False,
        medianprops=medianprops,
        whiskerprops=boxprops,
        boxprops=boxprops,
    )

    ax.tick_params(labelsize=13)
    return ax
