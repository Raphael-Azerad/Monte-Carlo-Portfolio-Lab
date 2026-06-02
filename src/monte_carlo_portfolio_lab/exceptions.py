"""Custom exceptions for clear app and test behavior."""


class DataProviderError(RuntimeError):
    """Raised when market data cannot be loaded or validated."""


class PortfolioValidationError(ValueError):
    """Raised when portfolio inputs are invalid."""


class SimulationError(RuntimeError):
    """Raised when a simulation cannot be completed."""
