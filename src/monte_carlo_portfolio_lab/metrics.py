"""Risk and performance metrics for simulated portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monte_carlo_portfolio_lab.config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from monte_carlo_portfolio_lab.simulation import SimulationResult


def summarize_simulation(
    result: SimulationResult,
    target_return: float,
    benchmark_result: SimulationResult | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, float]:
    """Calculate risk metrics from simulated path-level outcomes.

    Terminal distribution metrics are calculated across simulation outcomes.
    Volatility, Sharpe, Sortino, and drawdown are calculated for each simulated
    path first, then summarized across paths. This avoids understating risk by
    measuring only the volatility of the average simulated path.
    """

    terminal_values = result.terminal_values
    total_returns = terminal_values / result.initial_capital - 1
    annualized_returns = annualized_path_returns(result)
    path_volatility = annualized_path_volatility(result.daily_returns)
    downside_volatility = annualized_downside_volatility(result.daily_returns)
    path_sharpe = (annualized_returns - risk_free_rate) / path_volatility.replace(
        0, np.nan
    )
    path_sortino = (annualized_returns - risk_free_rate) / downside_volatility.replace(
        0, np.nan
    )
    drawdowns = maximum_drawdown(result.paths)
    var_95 = float(np.percentile(total_returns, 5))
    cvar_95 = float(total_returns[total_returns <= var_95].mean())

    metrics = {
        "expected_terminal_value": float(terminal_values.mean()),
        "median_terminal_value": float(terminal_values.median()),
        "p05_terminal_value": float(np.percentile(terminal_values, 5)),
        "p25_terminal_value": float(np.percentile(terminal_values, 25)),
        "p75_terminal_value": float(np.percentile(terminal_values, 75)),
        "p95_terminal_value": float(np.percentile(terminal_values, 95)),
        "p05_terminal_return": float(np.percentile(total_returns, 5)),
        "p95_terminal_return": float(np.percentile(total_returns, 95)),
        "probability_of_profit": float(
            (terminal_values > result.initial_capital).mean()
        ),
        "probability_of_loss": float((terminal_values < result.initial_capital).mean()),
        "probability_target_return": float((total_returns >= target_return).mean()),
        "annualized_return": float(annualized_returns.median()),
        "mean_annualized_return": float(annualized_returns.mean()),
        "annualized_volatility": float(path_volatility.median()),
        "average_path_volatility": float(path_volatility.mean()),
        "downside_volatility": float(downside_volatility.median()),
        "maximum_drawdown": float(drawdowns.mean()),
        "average_maximum_drawdown": float(drawdowns.mean()),
        "median_maximum_drawdown": float(drawdowns.median()),
        "p05_maximum_drawdown": float(np.percentile(drawdowns, 5)),
        "sharpe_ratio": _clean_median(path_sharpe),
        "sortino_ratio": _clean_median(path_sortino),
        "value_at_risk_95": var_95,
        "conditional_value_at_risk_95": cvar_95,
    }
    if benchmark_result is not None:
        benchmark_returns = (
            benchmark_result.terminal_values / benchmark_result.initial_capital - 1
        )
        benchmark_median = float(benchmark_returns.median())
        metrics["benchmark_median_total_return"] = benchmark_median
        metrics["probability_outperforming_benchmark_median"] = float(
            (total_returns > benchmark_median).mean()
        )
        metrics["probability_outperforming_benchmark_pathwise"] = float(
            (total_returns.to_numpy() > benchmark_returns.to_numpy()).mean()
        )
        metrics["probability_outperforming_benchmark"] = metrics[
            "probability_outperforming_benchmark_median"
        ]
    return metrics


def annualized_path_returns(result: SimulationResult) -> pd.Series:
    """Calculate annualized total return for each simulated path."""

    total_returns = result.terminal_values / result.initial_capital
    return total_returns ** (TRADING_DAYS_PER_YEAR / len(result.paths)) - 1


def annualized_path_volatility(daily_returns: pd.DataFrame) -> pd.Series:
    """Calculate annualized volatility separately for each simulated path."""

    return daily_returns.std(axis=0, ddof=1).fillna(0.0) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )


def annualized_downside_volatility(daily_returns: pd.DataFrame) -> pd.Series:
    """Calculate annualized downside volatility for each simulated path."""

    downside_returns = daily_returns.where(daily_returns < 0)
    return downside_returns.std(axis=0, ddof=1).fillna(0.0) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )


def maximum_drawdown(paths: pd.DataFrame) -> pd.Series:
    """Calculate maximum drawdown for each simulated path."""

    running_max = paths.cummax()
    drawdowns = paths / running_max - 1
    return drawdowns.min(axis=0)


def _clean_median(values: pd.Series) -> float:
    """Return a finite median for path-level ratio metrics."""

    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 0.0
    return float(finite.median())
