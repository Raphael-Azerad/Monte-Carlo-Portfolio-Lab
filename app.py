"""Streamlit app for Monte Carlo Portfolio Lab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from monte_carlo_portfolio_lab.config import (
    DEFAULT_CONFIG,
    DEFAULT_PORTFOLIO,
    SCENARIOS,
)
from monte_carlo_portfolio_lab.data import YFinanceDataProvider
from monte_carlo_portfolio_lab.exceptions import (
    DataProviderError,
    PortfolioValidationError,
    SimulationError,
)
from monte_carlo_portfolio_lab.metrics import summarize_simulation
from monte_carlo_portfolio_lab.portfolio import Portfolio
from monte_carlo_portfolio_lab.returns import calculate_log_returns, correlation_matrix
from monte_carlo_portfolio_lab.simulation import (
    HistoricalBootstrapEngine,
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

st.set_page_config(
    page_title="Monte Carlo Portfolio Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Render the Streamlit application."""

    st.title("Monte Carlo Portfolio Lab")
    st.caption(
        "Scenario analysis for terminal wealth distributions, downside risk, "
        "and benchmark-relative outcomes. Results are simulations, not forecasts."
    )

    inputs = render_sidebar()
    try:
        portfolio = Portfolio.from_mapping(inputs["weights"], inputs["capital"])
        if portfolio.normalized:
            st.warning("Weights were normalized automatically to sum to 100%.")

        all_tickers = sorted(set(portfolio.tickers + [inputs["benchmark"]]))
        with st.spinner("Loading market data and running simulations..."):
            prices = load_prices(all_tickers, inputs["lookback_years"])
            returns = calculate_log_returns(prices)
            result = run_simulation(returns, portfolio, inputs)
            benchmark = Portfolio.from_mapping(
                {inputs["benchmark"]: 1.0}, inputs["capital"]
            )
            benchmark_result = run_simulation(returns, benchmark, inputs)
            metrics = summarize_simulation(
                result,
                inputs["target_return"],
                benchmark_result=benchmark_result,
            )

        render_metric_cards(metrics)
        render_results(result, returns[portfolio.tickers], metrics, portfolio)
        render_comparison_tab(returns, inputs)
        render_methodology()

    except (
        DataProviderError,
        PortfolioValidationError,
        SimulationError,
        ValueError,
    ) as exc:
        st.error(str(exc))
        st.stop()


def render_sidebar() -> dict[str, object]:
    """Collect user inputs from the sidebar."""

    st.sidebar.header("Portfolio")
    default_tickers = ", ".join(DEFAULT_PORTFOLIO.keys())
    tickers_text = st.sidebar.text_input("Tickers", value=default_tickers)
    tickers = [
        ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip()
    ]
    default_weight_text = ", ".join(
        str(int(weight * 100)) for weight in DEFAULT_PORTFOLIO.values()
    )
    weights_text = st.sidebar.text_input("Weights (%)", value=default_weight_text)
    weights = _parse_weights(tickers, weights_text)

    capital = st.sidebar.number_input(
        "Initial capital ($)",
        min_value=100.0,
        value=float(DEFAULT_CONFIG.initial_capital),
        step=500.0,
    )
    benchmark = st.sidebar.text_input(
        "Benchmark", value=DEFAULT_CONFIG.benchmark
    ).upper()

    st.sidebar.header("Simulation")
    engine = st.sidebar.selectbox(
        "Engine",
        ["Parametric Monte Carlo", "Historical Bootstrap"],
        help=(
            "Parametric Monte Carlo samples correlated normal returns. Historical "
            "Bootstrap resamples observed return days from the lookback window."
        ),
    )
    simulations = st.sidebar.slider(
        "Simulation count",
        min_value=1_000,
        max_value=25_000,
        value=DEFAULT_CONFIG.simulations,
        step=1_000,
    )
    horizon = st.sidebar.slider(
        "Horizon (trading days)",
        min_value=21,
        max_value=756,
        value=DEFAULT_CONFIG.horizon_days,
        step=21,
    )
    lookback_years = st.sidebar.slider(
        "Historical lookback (years)",
        min_value=1,
        max_value=15,
        value=DEFAULT_CONFIG.lookback_years,
    )
    seed = st.sidebar.number_input(
        "Random seed", min_value=0, value=DEFAULT_CONFIG.seed
    )
    target_return = st.sidebar.slider(
        "Target return",
        min_value=-0.50,
        max_value=1.00,
        value=DEFAULT_CONFIG.target_return,
        step=0.01,
        format="%.0f%%",
    )

    st.sidebar.header("Scenario")
    scenario_name = st.sidebar.selectbox("Scenario", list(SCENARIOS.keys()))
    scenario = SCENARIOS[scenario_name]
    volatility_multiplier = st.sidebar.slider(
        "Volatility multiplier",
        min_value=0.25,
        max_value=3.00,
        value=float(scenario["volatility_multiplier"]),
        step=0.05,
        help="Scales daily return volatility before simulation.",
    )
    mean_shift = st.sidebar.slider(
        "Annual mean return shock",
        min_value=-0.30,
        max_value=0.30,
        value=float(scenario["mean_shift"]),
        step=0.01,
        format="%.0f%%",
        help="Annualized return shock applied to the historical mean estimate.",
    )
    correlation_stress = st.sidebar.slider(
        "Correlation stress",
        min_value=0.0,
        max_value=0.75,
        value=float(scenario["correlation_stress"]),
        step=0.05,
        help="Pushes asset correlations closer to 1.0 in stress scenarios.",
    )

    return {
        "weights": weights,
        "capital": capital,
        "benchmark": benchmark,
        "engine": engine,
        "simulations": simulations,
        "horizon": horizon,
        "lookback_years": lookback_years,
        "seed": int(seed),
        "target_return": target_return,
        "scenario": scenario_name,
        "volatility_multiplier": volatility_multiplier,
        "mean_shift": mean_shift,
        "correlation_stress": correlation_stress,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers: list[str], years: int) -> pd.DataFrame:
    """Load market prices with Streamlit caching."""

    return YFinanceDataProvider().get_price_history(tickers, years)


def run_simulation(
    returns: pd.DataFrame, portfolio: Portfolio, inputs: dict[str, object]
):
    """Run the selected simulation engine."""

    config = SimulationConfig(
        horizon_days=int(inputs["horizon"]),
        simulations=int(inputs["simulations"]),
        seed=int(inputs["seed"]),
        volatility_multiplier=float(inputs["volatility_multiplier"]),
        mean_shift=float(inputs["mean_shift"]),
        correlation_stress=float(inputs["correlation_stress"]),
    )
    engine = (
        ParametricMonteCarloEngine()
        if inputs["engine"] == "Parametric Monte Carlo"
        else HistoricalBootstrapEngine()
    )
    return engine.run(returns, portfolio, config)


def render_metric_cards(metrics: dict[str, float]) -> None:
    """Render the main metric dashboard."""

    st.subheader("Simulation Summary")
    st.caption(
        "Metrics summarize simulated path outcomes. Volatility, Sharpe, Sortino, "
        "and drawdown are calculated per path before being summarized."
    )
    columns = st.columns(4)
    card_data = [
        ("Expected Value", _money(metrics["expected_terminal_value"])),
        ("Median Value", _money(metrics["median_terminal_value"])),
        ("5th Percentile", _money(metrics["p05_terminal_value"])),
        ("Profit Probability", _percent(metrics["probability_of_profit"])),
        ("Target Probability", _percent(metrics["probability_target_return"])),
        (
            "Beat SPY Median",
            _percent(metrics["probability_outperforming_benchmark_median"]),
        ),
        ("Median Volatility", _percent(metrics["annualized_volatility"])),
        ("Median Drawdown", _percent(metrics["median_maximum_drawdown"])),
    ]
    for index, (label, value) in enumerate(card_data):
        columns[index % 4].metric(label, value)


def render_results(
    result,
    returns: pd.DataFrame,
    metrics: dict[str, float],
    portfolio: Portfolio,
) -> None:
    """Render chart tabs and supporting tables."""

    tab_dashboard, tab_risk, tab_data = st.tabs(["Results", "Risk", "Data"])
    with tab_dashboard:
        st.caption(
            "The fan chart and histogram show simulated terminal distributions. "
            "They are scenario estimates, not price targets."
        )
        st.plotly_chart(fan_chart(result), use_container_width=True)
        left, right = st.columns(2)
        left.plotly_chart(terminal_histogram(result), use_container_width=True)
        right.plotly_chart(allocation_bar(portfolio.weights), use_container_width=True)

    with tab_risk:
        st.caption(
            "Drawdown and correlation views focus on downside behavior and "
            "asset co-movement under the selected scenario."
        )
        left, right = st.columns(2)
        left.plotly_chart(drawdown_chart(result), use_container_width=True)
        right.plotly_chart(max_drawdown_distribution(result), use_container_width=True)
        st.plotly_chart(
            correlation_heatmap(correlation_matrix(returns)), use_container_width=True
        )

    with tab_data:
        metric_table = pd.Series(metrics).rename("value").to_frame()
        st.dataframe(metric_table, use_container_width=True)
        st.dataframe(
            portfolio.weights.rename("normalized_weight").to_frame(),
            use_container_width=True,
        )


def render_comparison_tab(returns: pd.DataFrame, inputs: dict[str, object]) -> None:
    """Render a lightweight Portfolio A/B/C comparison section."""

    st.subheader("Allocation Comparison")
    variants = {
        "Portfolio A": {"SPY": 0.40, "QQQ": 0.25, "TLT": 0.25, "GLD": 0.10},
        "Portfolio B": {"SPY": 0.60, "QQQ": 0.20, "TLT": 0.10, "GLD": 0.10},
        "Portfolio C": {"SPY": 0.25, "QQQ": 0.25, "TLT": 0.35, "GLD": 0.15},
    }
    rows = []
    for name, weights in variants.items():
        tickers = [ticker for ticker in weights if ticker in returns.columns]
        if len(tickers) != len(weights):
            continue
        portfolio = Portfolio.from_mapping(weights, float(inputs["capital"]))
        comparison = run_simulation(returns, portfolio, inputs)
        summary = summarize_simulation(comparison, float(inputs["target_return"]))
        rows.append(
            {
                "Portfolio": name,
                "Expected Value": summary["expected_terminal_value"],
                "5th Percentile": summary["p05_terminal_value"],
                "Profit Probability": summary["probability_of_profit"],
                "Median Annual Volatility": summary["annualized_volatility"],
                "Sharpe Ratio": summary["sharpe_ratio"],
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Comparison portfolios load when their default tickers are available.")


def render_methodology() -> None:
    """Render educational methodology and limitations."""

    with st.expander("Methodology"):
        st.write(
            "The parametric engine estimates daily log-return means and covariance "
            "from historical prices, then samples correlated returns using a "
            "multivariate normal model. The bootstrap engine resamples historical "
            "return rows to preserve more of the empirical distribution shape. "
            "Risk metrics are calculated at the simulated-path level before they "
            "are summarized across the distribution."
        )
    with st.expander("Limitations"):
        st.write(
            "Historical returns do not predict the future. Monte Carlo simulations "
            "are scenario-based and are not investment advice. Financial markets are "
            "not perfectly normally distributed, and tail events can exceed modeled "
            "expectations."
        )


def _parse_weights(tickers: list[str], weights_text: str) -> dict[str, float]:
    values = [
        float(value.strip()) / 100 for value in weights_text.split(",") if value.strip()
    ]
    if len(values) != len(tickers):
        raise PortfolioValidationError("Enter one weight for each ticker.")
    return dict(zip(tickers, values, strict=True))


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _percent(value: float) -> str:
    return f"{value:.1%}"


if __name__ == "__main__":
    main()
