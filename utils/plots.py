# File:         plots.py
# Date:         2024/07/01
# Description:  Contains functions to create plots

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px


def plot_errors(err: np.ndarray) -> go.Figure:
    """Plot the RMSE errors for position and orientation

    Parameters
    ----------
    err : np.ndarray
        Rollout errors ; shape [experiments, start_times, error_type, timesteps]

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            "Translation RMSE [m] across rollout timesteps",
            "Orientation RMSE [deg] across rollout timesteps",
        ],
        vertical_spacing=0.1,
    )

    # Plot all errors together
    timesteps = np.arange(0, err.shape[-1])
    mean_errors = np.mean(err, axis=(0, 1))
    std_errors = np.std(err, axis=(0, 1))

    # Plot the RMSE of all experiments
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=mean_errors[1],  # Positional RMSE
            mode="lines",
            name="Translation RMSE",
            line=dict(color=px.colors.qualitative.Plotly[0]),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.hstack([timesteps, timesteps[::-1]]),
            y=np.hstack(
                [
                    mean_errors[1] + std_errors[1] / 2,
                    (mean_errors[1] - std_errors[1] / 2)[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.2)",
            hoverinfo="skip",
            line=dict(color="rgba(255, 255, 255, 0)"),
            showlegend=True,
            name="Translation std. dev. (test set)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=mean_errors[3],  # Orientation RMSE
            mode="lines",
            name="Orientation RMSE",
            line=dict(color=px.colors.qualitative.Plotly[1]),
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.hstack([timesteps, timesteps[::-1]]),
            y=np.hstack(
                [
                    mean_errors[3] + std_errors[3] / 2,
                    (mean_errors[3] - std_errors[3] / 2)[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(239, 85, 59, 0.2)",
            hoverinfo="skip",
            line=dict(color="rgba(255, 255, 255, 0)"),
            showlegend=True,
            name="Orientation std. dev. (test set)",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Autoregressive Rollout Step", row=1, col=1)
    fig.update_xaxes(title_text="Autoregressive Rollout Step", row=2, col=1)
    fig.update_yaxes(title_text="Translation RMSE [m]", row=1, col=1)
    fig.update_yaxes(title_text="Orientation RMSE [deg]", row=2, col=1)

    fig.update_layout(
        font_family="Arial",
        height=980,
        width=1920,
        title_text="<b>Auto-Regressive Rollout Errors</b>",
        title_x=0.5,
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top", orientation="v"),
    )

    return fig


def plot_errors_distribution(err: np.ndarray) -> go.Figure:
    """Plot RMSE distribution

    Parameters
    ----------
    err : np.ndarray
        Rollout errors ; shape [experiments, start_times, error_type, timesteps]

    Returns
    -------
    go.Figure
        Plotly figure
    """
    CHECKPOINT_TIMES: list[int] = [25, 50, 75, 100]

    fig = make_subplots(
        rows=2,
        cols=len(CHECKPOINT_TIMES),
        subplot_titles=[f"After {s} steps" for s in CHECKPOINT_TIMES],
        vertical_spacing=0.1,
    )

    # Plot all errors together
    for col, time in enumerate(CHECKPOINT_TIMES):
        fig.add_trace(
            go.Histogram(
                x=err[:, :, 1, time].flatten(),
                name="Position RMSE",
                showlegend=False,
                nbinsx=10,
                marker=dict(color=px.colors.qualitative.Plotly[col]),
            ),
            row=1,
            col=col + 1,
        )
        fig.add_trace(
            go.Histogram(
                x=err[:, :, 3, time].flatten(),
                name="Orientation RMSE",
                showlegend=False,
                nbinsx=10,
                marker=dict(color=px.colors.qualitative.Plotly[col]),
            ),
            row=2,
            col=col + 1,
        )
        fig.update_xaxes(
            title_text="Position RMSE [m]", row=1, col=col + 1, range=[0, 0.3]
        )
        fig.update_xaxes(
            title_text="Orientation RMSE [deg]", row=2, col=col + 1, range=[0, 18.0]
        )
        fig.update_yaxes(title_text="Frequency", row=2, col=col + 1)

    fig.update_layout(
        height=980,
        width=1920,
        title_text="Distribution of Auto-Regressive Rollout Errors",
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig
