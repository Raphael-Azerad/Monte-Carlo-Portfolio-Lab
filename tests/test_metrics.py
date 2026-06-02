from __future__ import annotations

import pandas as pd
import pytest

from monte_carlo_portfolio_lab.metrics import (
    annualized_downside_volatility,
    annualized_path_volatility,
    maximum_drawdown,
    summarize_simulation,
)
from monte_carlo_portfolio_lab.portfolio import Portfolio
from monte_carlo_portfolio_lab.returns import calculate_log_returns
from monte_carlo_portfolio_lab.simulation import (
    ParametricMonteCarloEngine,
    SimulationConfig,
)


def test_summary_metrics_include_probabilities(sample_prices):
    returns = calculate_log_returns(sample_prices)
    portfolio = Portfolio.from_mapping({"SPY": 0.5, "QQQ": 0.5}, 10_000)
    config = SimulationConfig(horizon_days=30, simulations=200, seed=21)
    result = ParametricMonteCarloEngine().run(returns, portfolio, config)

    metrics = summarize_simulation(result, target_return=0.05)

    assert 0 <= metrics["probability_of_profit"] <= 1
    assert 0 <= metrics["probability_of_loss"] <= 1
    assert "conditional_value_at_risk_95" in metrics
    assert "median_maximum_drawdown" in metrics
    assert "p05_terminal_return" in metrics
    assert "average_path_volatility" in metrics


def test_maximum_drawdown_is_non_positive(sample_prices):
    returns = calculate_log_returns(sample_prices)
    portfolio = Portfolio.from_mapping({"SPY": 1.0}, 10_000)
    config = SimulationConfig(horizon_days=20, simulations=50, seed=2)
    result = ParametricMonteCarloEngine().run(returns, portfolio, config)

    drawdowns = maximum_drawdown(result.paths)

    assert (drawdowns <= 0).all()


def test_path_level_volatility_uses_columns():
    daily_returns = pd.DataFrame(
        {
            "steady": [0.01, 0.01, 0.01, 0.01],
            "volatile": [0.03, -0.02, 0.04, -0.01],
        }
    )

    volatility = annualized_path_volatility(daily_returns)
    downside = annualized_downside_volatility(daily_returns)

    assert volatility["steady"] == pytest.approx(0.0)
    assert volatility["volatile"] > volatility["steady"]
    assert downside["volatile"] > downside["steady"]


def test_var_cvar_and_benchmark_comparison(sample_prices):
    returns = calculate_log_returns(sample_prices)
    portfolio = Portfolio.from_mapping({"SPY": 0.5, "QQQ": 0.5}, 10_000)
    benchmark = Portfolio.from_mapping({"SPY": 1.0}, 10_000)
    config = SimulationConfig(horizon_days=30, simulations=200, seed=21)
    engine = ParametricMonteCarloEngine()

    result = engine.run(returns, portfolio, config)
    benchmark_result = engine.run(returns, benchmark, config)
    metrics = summarize_simulation(
        result, target_return=0.05, benchmark_result=benchmark_result
    )

    assert metrics["conditional_value_at_risk_95"] <= metrics["value_at_risk_95"]
    assert 0 <= metrics["probability_outperforming_benchmark_median"] <= 1
    assert 0 <= metrics["probability_outperforming_benchmark_pathwise"] <= 1
