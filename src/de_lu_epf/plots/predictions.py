import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


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


def plot_metric_barplot(actual_series, pred_df, metric, title=None, ascending=True):
    metric_name = get_metric_name(metric=metric)

    scores = []
    for model_name in pred_df.columns:
        score = metric(actual_series, pred_df[model_name])
        scores.append((model_name, score))

    df_scores = pd.DataFrame(scores, columns=["model", "score"])
    df_scores = df_scores.sort_values("score", ascending=ascending)

    fig, ax = plt.subplots(figsize=(6, 4))

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


def plot_residual_violinplot(actual_series, pred_df, title=None):
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

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.violinplot(
        data=residual_df, x="model", y="residual", palette="tab10", hue="model", ax=ax
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Residual")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


# def _format_imf_label(name):
#     name_lower = name.lower()
#     if "resid" in name_lower:
#         return "Residual"
#     if name_lower.startswith("imf"):
#         suffix = name_lower.replace("imf", "")
#         return f"IMF {suffix}" if suffix.isdigit() else name.upper()
#     return name.upper()


# def plot_imf_predictions_interactive(pred_df, raw_df=None, title=None):
#     pred_names = list(pred_df.columns)
#     labels = [_format_imf_label(col) for col in pred_names]

#     if raw_df is not None:
#         raw_names = list(raw_df.columns)
#         if raw_names != pred_names:
#             raise ValueError(
#                 "raw_df and pred_df must have the same columns in the same order"
#             )

#     n_rows = len(pred_names)

#     fig = make_subplots(
#         rows=n_rows,
#         cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.02,
#         subplot_titles=labels,
#     )

#     for i, (col, label) in enumerate(zip(pred_names, labels), start=1):
#         if raw_df is not None:
#             fig.add_trace(
#                 go.Scatter(
#                     x=raw_df.index,
#                     y=raw_df[col].values,
#                     mode="lines",
#                     name="Actual",
#                     line=dict(color="black", width=1),
#                     hovertemplate=f"Time: %{{x}}<br>{label} Actual: %{{y}}<extra></extra>",
#                     showlegend=(i == 1),
#                     legendgroup="actual",
#                 ),
#                 row=i,
#                 col=1,
#             )

#         fig.add_trace(
#             go.Scatter(
#                 x=pred_df.index,
#                 y=pred_df[col].values,
#                 mode="lines",
#                 name="Predicted" if raw_df is not None else label,
#                 line=dict(width=1),
#                 hovertemplate=(
#                     f"Time: %{{x}}<br>{label} Predicted: %{{y}}<extra></extra>"
#                     if raw_df is not None
#                     else f"Time: %{{x}}<br>{label}: %{{y}}<extra></extra>"
#                 ),
#                 showlegend=(i == 1),
#                 legendgroup="predicted",
#             ),
#             row=i,
#             col=1,
#         )

#         fig.update_yaxes(title_text=label, row=i, col=1)

#     fig.update_layout(
#         title=title,
#         template="plotly_white",
#         hovermode="x unified",
#         height=max(220 * n_rows, 400),
#         xaxis_title="Datetime",
#         legend=dict(
#             itemclick="toggle",
#             itemdoubleclick="toggleothers",
#         ),
#     )

#     fig.update_xaxes(rangeslider_visible=True, row=n_rows, col=1)

#     return fig
