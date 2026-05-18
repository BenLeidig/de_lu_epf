import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


def get_metric_name(metric):
    return " ".join(part.capitalize() for part in metric.__name__.split("_"))


def plot_predictions_interactive(actual_series, pred_df, title=None):
    model_names = list(pred_df.columns)
    n_models = len(model_names)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual_series.index,
            y=actual_series.values,
            mode="lines",
            name="Actual",
            line=dict(color="black", width=1),
            hovertemplate="Time: %{x}<br>Actual: %{y}<extra></extra>",
            visible=True,
        )
    )

    for model_name in model_names:
        fig.add_trace(
            go.Scatter(
                x=pred_df.index,
                y=pred_df[model_name].values,
                mode="lines",
                name=model_name,
                line=dict(width=1),
                hovertemplate=f"Time: %{{x}}<br>{model_name}: %{{y}}<extra></extra>",
                visible=True,
            )
        )

    buttons = []

    buttons.append(
        dict(
            label="All Models",
            method="update",
            args=[{"visible": [True] * (n_models + 1)}],
        )
    )

    for i, model_name in enumerate(model_names):
        visible = [True] + [False] * n_models
        visible[i + 1] = True

        buttons.append(
            dict(
                label=model_name,
                method="update",
                args=[{"visible": visible}],
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Datetime",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.02,
                xanchor="left",
                y=1,
                yanchor="top",
            )
        ],
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
    )

    fig.update_xaxes(rangeslider_visible=True)

    return fig


def plot_residuals_interactive(actual_series, pred_df, title=None):
    residual_df = pred_df.sub(actual_series, axis=0).astype("float64")
    zero_series = actual_series * 0.0

    fig = plot_predictions_interactive(
        actual_series=zero_series,
        pred_df=residual_df,
        title=title,
    )

    fig.update_layout(yaxis_title="Residual")
    return fig


def plot_metric_barplot(
    actual_series,
    pred_df,
    metric,
    title=None,
    ascending=False,
    figsize=(12, 8),
    inline_text: bool = True,
):
    metric_name = get_metric_name(metric=metric)

    scores = []
    for model_name in pred_df.columns:
        score = metric(actual_series, pred_df[model_name])
        scores.append((model_name, score))

    df_scores = pd.DataFrame(scores, columns=["model", "score"])
    df_scores = df_scores.sort_values("score", ascending=ascending).head(10)

    fig, ax = plt.subplots(figsize=figsize)

    tableau_colors = plt.colormaps["tab10"].colors  # type: ignore
    colors = [tableau_colors[i % len(tableau_colors)] for i in range(len(df_scores))]

    bars = ax.bar(df_scores["model"], df_scores["score"], color=colors)

    y_min = min(0, df_scores["score"].min())
    y_max = max(0, df_scores["score"].max())
    y_range = y_max - y_min if y_max != y_min else 1

    if inline_text:
        for bar, score, model_name in zip(
            bars, df_scores["score"], df_scores["model"].unique()
        ):
            height = bar.get_height()
            offset = 0.03 * y_range

            if height >= 0:
                y1 = height - offset
                va1 = "top"
            else:
                y1 = height + offset
                va1 = "bottom"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y1,
                f"{score:.1f}",
                ha="center",
                va=va1,
                color="white",
                fontsize=18,
                fontweight="bold",
            )

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height // 2,
                model_name,
                ha="center",
                va="center",
                color="white",
                fontsize=24,
                fontweight="bold",
                rotation=90,
                rotation_mode="anchor",
            )
        ax.set_xticks([])
    else:
        for bar, score, model_name in zip(
            bars, df_scores["score"], df_scores["model"].unique()
        ):
            height = bar.get_height()
            offset = 0.03 * y_range

            if height >= 0:
                y1 = height - offset
                va1 = "top"
            else:
                y1 = height + offset
                va1 = "bottom"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y1,
                f"{score:.1f}",
                ha="center",
                va=va1,
                color="white",
                fontsize=18,
                fontweight="bold",
            )
        ax.tick_params(axis="x", rotation=-45)

    ax.tick_params(axis="y", labelsize=18)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.set_xlabel("Model", fontsize=18)
    ax.set_ylabel(metric_name, fontsize=18, fontweight="bold")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def plot_residual_violinplot(actual_series, pred_df, title=None, figsize=(12, 8)):
    mae_scores = {
        col: np.mean(np.abs(actual_series - pred_df[col])) for col in pred_df.columns
    }
    top_models = pd.Series(mae_scores).sort_values(ascending=True).head(10).index

    pred_df = pred_df[top_models]

    residual_df = (
        pred_df.sub(actual_series, axis=0)
        .astype("float64")
        .reset_index(names="datetime")
    )
    residual_df = residual_df.melt(
        id_vars="datetime",
        var_name="model",
        value_name="residual",
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=residual_df, x="model", y="residual", palette="tab10", hue="model", ax=ax
    )

    ax.tick_params(axis="x", labelsize=18)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.tick_params(axis="y", labelsize=18)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.set_xlabel("")
    ax.set_ylabel("Residuals", fontsize=18, fontweight="bold")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def plot_month_preds(month, year, actual_series, pred_df, title=None, figsize=(12, 8)):
    # mae_scores = {
    #     col: np.mean(np.abs(actual_series - pred_df[col])) for col in pred_df.columns
    # }
    # top_models = pd.Series(mae_scores).sort_values(ascending=True).head(10).index
    top_models = ["VTLM"]

    pred_df = pred_df[top_models]

    mask = (pred_df.index.month == month) & (pred_df.index.year == year)

    pred_df = pred_df[mask].reset_index(names="datetime")

    pred_df = pred_df.melt(
        id_vars="datetime",
        var_name="Model",
        value_name="forecast",
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(0, color="black", linewidth=0.7)

    ax.plot(
        actual_series[mask].index,
        actual_series[mask].values,
        label="Actual",
        c="black",
        linewidth=4,
    )

    ax.plot(
        pred_df["datetime"], pred_df["forecast"], label="VTLM", c="tab:red", linewidth=4
    )

    # sns.lineplot(
    #     data=pred_df,
    #     x="datetime",
    #     y="forecast",
    #     hue="Model",
    #     palette="tab10",
    #     ax=ax,
    #     linewidth=4,
    # )

    ax.tick_params(axis="x", labelsize=18)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.tick_params(axis="y", labelsize=18)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.legend(fontsize=18)
    ax.set_xlabel("")
    ax.set_ylabel("Spot Price (EUR/MWHr)", fontsize=18, fontweight="bold")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax
