"""Monte Carlo simulation engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from monte_carlo_portfolio_lab.exceptions import SimulationError
from monte_carlo_portfolio_lab.portfolio import Portfolio
from monte_carlo_portfolio_lab.scenarios import (
    apply_parametric_scenario,
    apply_return_scenario,
)


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation inputs shared by all engines."""

    horizon_days: int
    simulations: int
    seed: int = 42
    volatility_multiplier: float = 1.0
    mean_shift: float = 0.0
    correlation_stress: float = 0.0


@dataclass(frozen=True)
class SimulationResult:
    """Container for simulated portfolio paths."""

    paths: pd.DataFrame
    daily_returns: pd.DataFrame
    engine_name: str
    config: SimulationConfig
    initial_capital: float

    @property
    def terminal_values(self) -> pd.Series:
        """Final simulated portfolio values."""

        return self.paths.iloc[-1]


class SimulationEngine(Protocol):
    """Protocol for portfolio simulation engines."""

    name: str

    def run(
        self, returns: pd.DataFrame, portfolio: Portfolio, config: SimulationConfig
    ) -> SimulationResult:
        """Run a simulation and return portfolio paths."""


class ParametricMonteCarloEngine:
    """Monte Carlo engine using correlated multivariate normal log returns."""

    name = "Parametric Monte Carlo"

    def run(
        self, returns: pd.DataFrame, portfolio: Portfolio, config: SimulationConfig
    ) -> SimulationResult:
        """Simulate future portfolio values from mean and covariance estimates."""

        aligned = returns.reindex(columns=portfolio.tickers).dropna(how="any")
        mean_returns, cov = apply_parametric_scenario(
            aligned.mean(),
            aligned.cov(),
            config.volatility_multiplier,
            config.mean_shift,
            config.correlation_stress,
        )
        rng = np.random.default_rng(config.seed)
        try:
            samples = rng.multivariate_normal(
                mean_returns.to_numpy(),
                cov.to_numpy(),
                size=(config.horizon_days, config.simulations),
                check_valid="raise",
            )
        except ValueError as exc:
            raise SimulationError(
                "Covariance matrix is not usable for simulation."
            ) from exc

        weighted_log_returns = samples @ portfolio.weights.to_numpy()
        return _build_result(weighted_log_returns, portfolio, config, self.name)


class HistoricalBootstrapEngine:
    """Monte Carlo engine using resampled historical asset returns."""

    name = "Historical Bootstrap"

    def run(
        self, returns: pd.DataFrame, portfolio: Portfolio, config: SimulationConfig
    ) -> SimulationResult:
        """Simulate future values by resampling historical return rows."""

        aligned = returns.reindex(columns=portfolio.tickers).dropna(how="any")
        stressed = apply_return_scenario(
            aligned,
            config.volatility_multiplier,
            config.mean_shift,
        )
        rng = np.random.default_rng(config.seed)
        row_index = rng.integers(
            0, len(stressed), size=(config.horizon_days, config.simulations)
        )
        sampled = stressed.to_numpy()[row_index]
        weighted_log_returns = sampled @ portfolio.weights.to_numpy()
        return _build_result(weighted_log_returns, portfolio, config, self.name)


def _build_result(
    weighted_log_returns: np.ndarray,
    portfolio: Portfolio,
    config: SimulationConfig,
    engine_name: str,
) -> SimulationResult:
    """Convert simulated daily log returns into value paths."""

    growth = np.exp(np.cumsum(weighted_log_returns, axis=0))
    values = portfolio.initial_capital * growth
    index = pd.RangeIndex(1, config.horizon_days + 1, name="day")
    columns = pd.RangeIndex(1, config.simulations + 1, name="simulation")
    return SimulationResult(
        paths=pd.DataFrame(values, index=index, columns=columns),
        daily_returns=pd.DataFrame(
            np.exp(weighted_log_returns) - 1, index=index, columns=columns
        ),
        engine_name=engine_name,
        config=config,
        initial_capital=portfolio.initial_capital,
    )
