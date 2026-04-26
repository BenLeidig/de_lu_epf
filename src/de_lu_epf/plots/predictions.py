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
    actual_series, pred_df, metric, title=None, ascending=True, figsize=(6, 4)
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

    for bar, score in zip(bars, df_scores["score"]):
        height = bar.get_height()
        offset = 0.03 * y_range

        if height >= 0:
            y = height - offset
            va = "top"
        else:
            y = height + offset
            va = "bottom"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{score:.3f}",
            ha="center",
            va=va,
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def plot_residual_violinplot(actual_series, pred_df, title=None, figsize=(6, 4)):
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

    ax.set_xlabel("Model")
    ax.set_ylabel("Residual")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def plot_month_preds(month, year, actual_series, pred_df, title=None, figsize=(6, 4)):
    mae_scores = {
        col: np.mean(np.abs(actual_series - pred_df[col])) for col in pred_df.columns
    }
    top_models = pd.Series(mae_scores).sort_values(ascending=True).head(10).index

    pred_df = pred_df[top_models]
    pred_df = pred_df[
        (pred_df.index.month == month) & (pred_df.index.year == year)
    ].reset_index(names="datetime")
    pred_df = pred_df.melt(
        id_vars="datetime",
        var_name="model",
        value_name="forecast",
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=pred_df, x=pred_df.index, y="forecast", hue="model", palette="tab10", ax=ax
    )

    ax.set_xlabel("Datetime")
    ax.set_ylabel("Spot Price (EUR/MWHr)")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax