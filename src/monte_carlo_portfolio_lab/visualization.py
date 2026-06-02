"""Plotly chart builders used by the Streamlit application."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from monte_carlo_portfolio_lab.metrics import maximum_drawdown
from monte_carlo_portfolio_lab.simulation import SimulationResult


def fan_chart(result: SimulationResult) -> go.Figure:
    """Create a percentile fan chart for simulated portfolio paths."""

    percentiles = result.paths.quantile([0.05, 0.25, 0.50, 0.75, 0.95], axis=1).T
    fig = go.Figure()
    fig.add_trace(
        _line(percentiles.index, percentiles[0.95], "95th percentile", "#9DB7B3")
    )
    fig.add_trace(
        _line(percentiles.index, percentiles[0.75], "75th percentile", "#6F9993")
    )
    fig.add_trace(
        _line(percentiles.index, percentiles[0.50], "Median path", "#183B3A", 3)
    )
    fig.add_trace(
        _line(percentiles.index, percentiles[0.25], "25th percentile", "#C28E5C")
    )
    fig.add_trace(
        _line(percentiles.index, percentiles[0.05], "5th percentile", "#B44D43")
    )
    fig.update_layout(
        title="Simulated Portfolio Value Fan Chart",
        xaxis_title="Trading day",
        yaxis_title="Portfolio value",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def terminal_histogram(result: SimulationResult) -> go.Figure:
    """Create a terminal wealth histogram with percentile markers."""

    terminal = result.terminal_values
    fig = px.histogram(
        terminal,
        nbins=60,
        title="Distribution of Ending Portfolio Values",
        labels={"value": "Terminal value"},
        template="plotly_white",
        color_discrete_sequence=["#2F6F6D"],
    )
    for percentile, color in [(5, "#B44D43"), (50, "#183B3A"), (95, "#6F9993")]:
        fig.add_vline(
            x=terminal.quantile(percentile / 100),
            line_dash="dash",
            line_color=color,
            annotation_text=f"{percentile}th",
        )
    fig.update_layout(showlegend=False, yaxis_title="Simulation count")
    return fig


def correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    """Create a correlation heatmap for portfolio assets."""

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="Asset Correlation Matrix",
        template="plotly_white",
    )
    return fig


def drawdown_chart(result: SimulationResult) -> go.Figure:
    """Create a chart of median, 5th, and 95th percentile drawdown paths."""

    running_max = result.paths.cummax()
    drawdowns = result.paths / running_max - 1
    bands = drawdowns.quantile([0.05, 0.50, 0.95], axis=1).T
    fig = go.Figure()
    fig.add_trace(
        _line(bands.index, bands[0.95], "95th percentile drawdown", "#9DB7B3")
    )
    fig.add_trace(_line(bands.index, bands[0.50], "Median drawdown", "#183B3A", 3))
    fig.add_trace(_line(bands.index, bands[0.05], "5th percentile drawdown", "#B44D43"))
    fig.update_layout(
        title="Simulated Drawdown Paths",
        xaxis_title="Trading day",
        yaxis_title="Drawdown",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def allocation_bar(
    weights: pd.Series, title: str = "Portfolio Allocation"
) -> go.Figure:
    """Create an allocation bar chart."""

    fig = px.bar(
        x=weights.index,
        y=weights.values,
        title=title,
        labels={"x": "Ticker", "y": "Weight"},
        template="plotly_white",
        color=weights.index,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickformat=".0%")
    return fig


def max_drawdown_distribution(result: SimulationResult) -> go.Figure:
    """Create a distribution chart for maximum drawdown."""

    drawdowns = maximum_drawdown(result.paths)
    fig = px.histogram(
        drawdowns,
        nbins=50,
        title="Maximum Drawdown Distribution",
        labels={"value": "Maximum drawdown"},
        template="plotly_white",
        color_discrete_sequence=["#B44D43"],
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    return fig


def _line(
    x: pd.Index,
    y: pd.Series,
    name: str,
    color: str,
    width: int = 2,
) -> go.Scatter:
    """Build a standard Plotly line trace."""

    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line={"color": color, "width": width},
    )
