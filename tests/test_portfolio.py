from __future__ import annotations

import pytest

from monte_carlo_portfolio_lab.exceptions import PortfolioValidationError
from monte_carlo_portfolio_lab.portfolio import Portfolio, normalize_weights


def test_weight_normalization():
    portfolio = Portfolio.from_mapping({"SPY": 40, "QQQ": 60}, 10_000)

    assert portfolio.normalized is True
    assert portfolio.weights.sum() == pytest.approx(1.0)
    assert portfolio.weights["SPY"] == pytest.approx(0.4)


def test_negative_weights_raise_error():
    with pytest.raises(PortfolioValidationError):
        Portfolio.from_mapping({"SPY": 1.2, "TLT": -0.2}, 10_000)


def test_zero_weight_sum_raises_error():
    with pytest.raises(PortfolioValidationError):
        normalize_weights(Portfolio.from_mapping({"SPY": 1}, 10_000).weights * 0)


def test_weighted_returns(sample_prices):
    portfolio = Portfolio.from_mapping({"SPY": 0.5, "QQQ": 0.5}, 10_000)
    returns = sample_prices[["SPY", "QQQ"]].pct_change().dropna()

    weighted = portfolio.weighted_returns(returns)

    assert len(weighted) == len(returns)
