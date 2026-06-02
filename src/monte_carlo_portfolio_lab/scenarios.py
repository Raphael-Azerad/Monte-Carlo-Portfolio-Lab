"""Scenario stress transformations for simulation inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_parametric_scenario(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    volatility_multiplier: float = 1.0,
    mean_shift: float = 0.0,
    correlation_stress: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply intuitive stress controls to mean returns and covariance."""

    stressed_mean = mean_returns + mean_shift / 252
    vol = np.sqrt(np.diag(covariance.to_numpy()))
    corr = covariance.to_numpy() / np.outer(vol, vol)
    corr = np.nan_to_num(corr, nan=0.0)
    if correlation_stress:
        corr = corr + (1 - corr) * correlation_stress
        np.fill_diagonal(corr, 1.0)
    stressed_vol = vol * volatility_multiplier
    stressed_cov = corr * np.outer(stressed_vol, stressed_vol)
    return stressed_mean, pd.DataFrame(
        stressed_cov, index=covariance.index, columns=covariance.columns
    )


def apply_return_scenario(
    returns: pd.DataFrame,
    volatility_multiplier: float = 1.0,
    mean_shift: float = 0.0,
) -> pd.DataFrame:
    """Stress historical returns for bootstrap simulation."""

    centered = returns - returns.mean()
    shifted = returns.mean() + mean_shift / 252
    return shifted + centered * volatility_multiplier
