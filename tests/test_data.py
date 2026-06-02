from __future__ import annotations

from datetime import date

import pytest

from monte_carlo_portfolio_lab.data import (
    YFinanceDataProvider,
    _extract_close_prices,
    clean_price_history,
)
from monte_carlo_portfolio_lab.exceptions import DataProviderError


def test_clean_price_history_forward_fills(sample_prices):
    prices = sample_prices.copy()
    prices.iloc[2, 0] = None

    cleaned = clean_price_history(prices, min_observations=20)

    assert not cleaned.isna().any().any()


def test_clean_price_history_rejects_short_history(sample_prices):
    with pytest.raises(DataProviderError):
        clean_price_history(sample_prices.head(5), min_observations=20)


def test_clean_price_history_rejects_non_positive(sample_prices):
    prices = sample_prices.copy()
    prices.iloc[10, 0] = 0

    with pytest.raises(DataProviderError):
        clean_price_history(prices, min_observations=20)


def test_extract_close_prices_from_single_ticker(sample_prices):
    raw = sample_prices[["SPY"]].rename(columns={"SPY": "Close"})

    prices = _extract_close_prices(raw, ["SPY"])

    assert list(prices.columns) == ["SPY"]


def test_yfinance_provider_uses_downloaded_data(monkeypatch, sample_prices):
    def fake_download(**kwargs):
        assert kwargs["start"] < kwargs["end"]
        return sample_prices

    monkeypatch.setattr("monte_carlo_portfolio_lab.data.yf.download", fake_download)
    provider = YFinanceDataProvider(min_observations=20)

    prices = provider.get_price_history(["SPY", "QQQ"], years=1, end=date(2025, 1, 1))

    assert list(prices.columns) == ["SPY", "QQQ"]


def test_yfinance_provider_rejects_empty_tickers():
    provider = YFinanceDataProvider()

    with pytest.raises(DataProviderError):
        provider.get_price_history([], years=1)
