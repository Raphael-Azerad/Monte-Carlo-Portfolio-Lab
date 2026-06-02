from __future__ import annotations

import numpy as np

from monte_carlo_portfolio_lab.returns import (
    calculate_log_returns,
    calculate_simple_returns,
    covariance_matrix,
)


def test_log_returns_drop_first_row(sample_prices):
    returns = calculate_log_returns(sample_prices)

    assert len(returns) == len(sample_prices) - 1
    assert np.isfinite(returns.to_numpy()).all()


def test_simple_returns_are_positive_for_rising_series(sample_prices):
    returns = calculate_simple_returns(sample_prices[["SPY", "TLT", "GLD"]])

    assert (returns > 0).all().all()


def test_covariance_matrix_shape(sample_prices):
    returns = calculate_log_returns(sample_prices)
    covariance = covariance_matrix(returns)

    assert covariance.shape == (4, 4)
    assert list(covariance.columns) == ["SPY", "QQQ", "TLT", "GLD"]
