"""Market data loading and cleaning utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from monte_carlo_portfolio_lab.exceptions import DataProviderError


class MarketDataProvider(ABC):
    """Interface for market data providers."""

    @abstractmethod
    def get_price_history(
        self, tickers: list[str], years: int, end: date | None = None
    ) -> pd.DataFrame:
        """Return adjusted close prices indexed by date."""


@dataclass
class YFinanceDataProvider(MarketDataProvider):
    """Market data provider backed by Yahoo Finance through yfinance."""

    min_observations: int = 60

    def get_price_history(
        self, tickers: list[str], years: int, end: date | None = None
    ) -> pd.DataFrame:
        """Download and validate adjusted close prices for the requested tickers."""

        if not tickers:
            raise DataProviderError("At least one ticker is required.")
        if years < 1:
            raise DataProviderError("Lookback period must be at least one year.")

        end_date = end or date.today()
        start_date = end_date - timedelta(days=int(years * 365.25) + 10)
        raw = yf.download(
            tickers=tickers,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            auto_adjust=True,
            group_by="column",
            threads=True,
        )
        prices = _extract_close_prices(raw, tickers)
        return clean_price_history(prices, self.min_observations)


def _extract_close_prices(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract close prices from the common yfinance response shapes."""

    if raw.empty:
        raise DataProviderError("No market data was returned.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise DataProviderError("Downloaded data does not contain close prices.")
        prices = raw["Close"]
    elif "Close" in raw.columns:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        prices = raw

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.reindex(columns=tickers)
    return prices


def clean_price_history(
    prices: pd.DataFrame, min_observations: int = 60
) -> pd.DataFrame:
    """Clean adjusted close price history and enforce data quality standards."""

    if prices.empty:
        raise DataProviderError("Price history is empty.")

    cleaned = prices.sort_index().ffill().dropna(axis=0, how="any")
    cleaned = cleaned.loc[:, cleaned.notna().any()]
    missing = [column for column in prices.columns if column not in cleaned.columns]
    if missing:
        raise DataProviderError(f"No usable price history for: {', '.join(missing)}")
    if len(cleaned) < min_observations:
        raise DataProviderError(
            f"Only {len(cleaned)} observations are available; "
            f"{min_observations} are required."
        )
    if (cleaned <= 0).any().any():
        raise DataProviderError("Price history contains non-positive prices.")
    return cleaned.astype(float)
