from __future__ import annotations

from monte_carlo_portfolio_lab.returns import calculate_log_returns
from monte_carlo_portfolio_lab.scenarios import apply_parametric_scenario


def test_scenario_volatility_multiplier_increases_covariance(sample_prices):
    returns = calculate_log_returns(sample_prices)
    mean, covariance = apply_parametric_scenario(
        returns.mean(),
        returns.cov(),
        volatility_multiplier=2.0,
        mean_shift=-0.05,
        correlation_stress=0.2,
    )

    assert len(mean) == len(returns.columns)
    assert covariance.loc["SPY", "SPY"] > returns.cov().loc["SPY", "SPY"]
