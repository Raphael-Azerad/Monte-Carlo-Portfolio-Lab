"""Application defaults and reusable constants."""

from dataclasses import dataclass

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02

DEFAULT_PORTFOLIO: dict[str, float] = {
    "SPY": 0.40,
    "QQQ": 0.25,
    "TLT": 0.25,
    "GLD": 0.10,
}


@dataclass(frozen=True)
class SimulationDefaults:
    """Default inputs used by the Streamlit application."""

    initial_capital: float = 10_000.0
    lookback_years: int = 5
    simulations: int = 10_000
    horizon_days: int = 252
    seed: int = 42
    benchmark: str = "SPY"
    target_return: float = 0.08


DEFAULT_CONFIG = SimulationDefaults()

SCENARIOS: dict[str, dict[str, float]] = {
    "Normal Market": {
        "volatility_multiplier": 1.0,
        "mean_shift": 0.0,
        "correlation_stress": 0.0,
    },
    "High Volatility Market": {
        "volatility_multiplier": 1.5,
        "mean_shift": -0.02,
        "correlation_stress": 0.15,
    },
    "Market Stress Test": {
        "volatility_multiplier": 2.0,
        "mean_shift": -0.08,
        "correlation_stress": 0.35,
    },
}
