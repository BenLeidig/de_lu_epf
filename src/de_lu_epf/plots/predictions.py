import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from de_lu_epf.plots.utils import plot_series, raincloudplot


def plot_predictions(
    index,
    actual_series_dict,
    pred_series_dict: dict,
    suptitle: str = None,  # type: ignore
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    colors = plt.colormaps["tab10"].colors  # type: ignore

    actual_label = list(actual_series_dict.keys())[0]
    actual_series = actual_series_dict[actual_label]
    plot_series(
        index=index,
        values=actual_series,
        ax=ax,
        color="black",
        label=actual_label,
        linewidth=0.5,
    )

    for i, (label, values) in enumerate(pred_series_dict.items()):
        color = colors[i % len(colors)]
        plot_series(
            index=index,
            values=values,
            ax=ax,
            color=color,
            label=label,
            axhline=False,
            linewidth=0.5,
        )

    ax.legend()
    if suptitle is not None:
        ax.set_title(suptitle)

    return fig, ax


def plot_residuals(
    index,
    actual_series,
    pred_series_dict: dict,
    suptitle: str = None,  # type: ignore
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    colors = plt.colormaps["tab10"].colors  # type: ignore

    for i, (label, values) in enumerate(pred_series_dict.items()):
        color = colors[i % len(colors)]
        residuals = actual_series - values
        plot_series(
            index=index,
            values=residuals,
            ax=ax,
            color=color,
            label=label,
            axhline=True if i == 0 else False,
            linewidth=0.5,
        )

    ax.legend()
    if suptitle is not None:
        ax.set_title(suptitle)

    return fig, ax


def plot_residuals_violinplot(
    actual_series,
    pred_series_dict: dict,
    suptitle: str = None,  # type: ignore
):
    fig, ax = plt.subplots(figsize=(6, 4))

    data = []
    for label, values in pred_series_dict.items():
        residuals = actual_series - values
        data.append(pd.DataFrame({"model": label, "residual": residuals}))
    df = pd.concat(data, ignore_index=True)

    palette = plt.colormaps["tab10"].colors  # type: ignore
    ax.axhline(0, color="black", linewidth=0.7)
    sns.violinplot(
        data=df,
        x="model",
        y="residual",
        ax=ax,
        inner="box",
        cut=0,
        palette=palette,
        hue="model",
        legend=False,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Residuals")
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()

    return fig, ax


def plot_residuals_barplot(
    actual_series,
    pred_series_dict: dict,
    metric,
    suptitle: str = None,  # type: ignore
):
    fig, ax = plt.subplots(figsize=(6, 4))

    metric_name = metric.__name__.split("_")
    metric_name = [m[0].upper() + m[1:] for m in metric_name]
    metric_name = " ".join(metric_name)

    data = []
    for label, values in pred_series_dict.items():
        score = metric(actual_series, values)
        data.append({"model": label, "score": score})
    df = pd.DataFrame(data)

    palette = plt.colormaps["tab10"].colors  # type: ignore
    sns.barplot(
        data=df,
        x="model",
        y="score",
        ax=ax,
        palette=palette,
        hue="model",
        legend=False,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()

    return fig, ax


def plot_residuals_raincloud(
    actual_series,
    pred_series_dict: dict,
    suptitle: str = None,  # type: ignore
):
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "model": model_name,
                    "residuals": actual_series - preds,
                }
            )
            for model_name, preds in pred_series_dict.items()
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    palette = plt.colormaps["tab10"].colors  # type: ignore

    ax = raincloudplot(x="residuals", y="model", palette=palette, data=df, ax=ax)

    ax.set_xlabel("Residuals")
    ax.set_ylabel("Model")

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    return fig, ax
