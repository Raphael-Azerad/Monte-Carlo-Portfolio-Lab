from __future__ import annotations

from monte_carlo_portfolio_lab.portfolio import Portfolio
from monte_carlo_portfolio_lab.returns import calculate_log_returns, correlation_matrix
from monte_carlo_portfolio_lab.simulation import (
    ParametricMonteCarloEngine,
    SimulationConfig,
)
from monte_carlo_portfolio_lab.visualization import (
    allocation_bar,
    correlation_heatmap,
    drawdown_chart,
    fan_chart,
    max_drawdown_distribution,
    terminal_histogram,
)


def _result(sample_prices):
    returns = calculate_log_returns(sample_prices)
    portfolio = Portfolio.from_mapping({"SPY": 0.5, "QQQ": 0.5}, 10_000)
    config = SimulationConfig(horizon_days=20, simulations=50, seed=12)
    return (
        ParametricMonteCarloEngine().run(returns, portfolio, config),
        returns,
        portfolio,
    )


def test_chart_builders_return_figures(sample_prices):
    result, returns, portfolio = _result(sample_prices)

    figures = [
        fan_chart(result),
        terminal_histogram(result),
        correlation_heatmap(correlation_matrix(returns)),
        drawdown_chart(result),
        allocation_bar(portfolio.weights),
        max_drawdown_distribution(result),
    ]

    assert all(figure.data for figure in figures)
