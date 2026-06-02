"""Return and covariance calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns from adjusted close prices."""

    returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    return returns.dropna(how="any")


def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily simple returns from adjusted close prices."""

    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Return the sample covariance matrix for asset returns."""

    return returns.cov()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Return the sample correlation matrix for asset returns."""

    return returns.corr()
