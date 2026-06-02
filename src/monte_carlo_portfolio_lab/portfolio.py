"""Portfolio construction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from monte_carlo_portfolio_lab.exceptions import PortfolioValidationError


@dataclass(frozen=True)
class Portfolio:
    """Long-only portfolio with normalized weights."""

    weights: pd.Series
    initial_capital: float
    normalized: bool = False

    @classmethod
    def from_mapping(
        cls, weights: dict[str, float], initial_capital: float
    ) -> Portfolio:
        """Build a portfolio from ticker-weight pairs."""

        tickers = [ticker.strip().upper() for ticker in weights if ticker.strip()]
        values = [float(weights[ticker]) for ticker in weights if ticker.strip()]
        series = pd.Series(values, index=tickers, dtype=float)
        normalized_weights, changed = normalize_weights(series)
        if initial_capital <= 0:
            raise PortfolioValidationError("Initial capital must be positive.")
        return cls(
            weights=normalized_weights,
            initial_capital=float(initial_capital),
            normalized=changed,
        )

    @property
    def tickers(self) -> list[str]:
        """Portfolio tickers in weight order."""

        return list(self.weights.index)

    def weighted_returns(self, asset_returns: pd.DataFrame) -> pd.Series:
        """Calculate weighted portfolio returns from asset returns."""

        aligned = asset_returns.reindex(columns=self.tickers).dropna(how="any")
        return aligned @ self.weights


def normalize_weights(weights: pd.Series) -> tuple[pd.Series, bool]:
    """Validate and normalize long-only weights to sum to one."""

    if weights.empty:
        raise PortfolioValidationError("At least one asset weight is required.")
    if weights.index.duplicated().any():
        raise PortfolioValidationError("Duplicate tickers are not allowed.")
    if weights.isna().any():
        raise PortfolioValidationError("Weights cannot be blank.")
    if (weights < 0).any():
        raise PortfolioValidationError("This version supports long-only portfolios.")

    total = float(weights.sum())
    if np.isclose(total, 0):
        raise PortfolioValidationError("Weights must sum to a positive value.")
    normalized = weights / total
    changed = not np.isclose(total, 1.0)
    return normalized, bool(changed)
