from __future__ import annotations

import pandas as pd

from monte_carlo_portfolio_lab.portfolio import Portfolio
from monte_carlo_portfolio_lab.returns import calculate_log_returns
from monte_carlo_portfolio_lab.simulation import (
    HistoricalBootstrapEngine,
    ParametricMonteCarloEngine,
    SimulationConfig,
)


def test_parametric_simulation_reproducible(sample_prices):
    returns = calculate_log_returns(sample_prices)
    portfolio = Portfolio.from_mapping({"SPY": 0.6, "QQQ": 0.4}, 10_000)
    config = SimulationConfig(horizon_days=30, simulations=100, seed=7)

    first = ParametricMonteCarloEngine().run(returns, portfolio, config)
    second = ParametricMonteCarloEngine().run(returns, portfolio, config)

    pd.testing.assert_frame_equal(first.paths, second.paths)


def test_bootstrap_simulation_shape(sample_prices):
    returns = calculate_log_returns(sample_prices)
    portfolio = Portfolio.from_mapping({"SPY": 0.4, "QQQ": 0.3, "TLT": 0.3}, 5_000)
    config = SimulationConfig(horizon_days=21, simulations=250, seed=11)

    result = HistoricalBootstrapEngine().run(returns, portfolio, config)

    assert result.paths.shape == (21, 250)
    assert result.terminal_values.gt(0).all()
