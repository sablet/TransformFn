"""Simulation and evaluation Transform functions for algorithmic trading pipeline."""

from __future__ import annotations

from typing import Annotated, List

from xform_core import Check, ExampleValue

import numpy as np
import pandas as pd

from xform_core import transform

from algo_trade_dtype.generators import (
    gen_prediction_data,
    gen_ranked_prediction_data,
    gen_selected_currency_data,
    gen_simulation_result,
)
from algo_trade_dtype.types import (
    PerformanceMetrics,
    PredictionData,
    RankedPredictionData,
    SelectedCurrencyData,
    SimulationResult,
)
from algo_trade_dtype.types import (
    PerformanceMetrics,
    PredictionData,
    RankedPredictionData,
    SelectedCurrencyData,
    SimulationResult,
)

_MAX_THRESHOLD_PCT = 0.5


@transform
def rank_predictions(
    predictions: Annotated[
        List[PredictionData],
        ExampleValue([
            {
                "date": "2024-01-01",
                "currency_pair": "USD_JPY",
                "prediction": 0.01,
                "actual_return": 0.005,
            },
            {
                "date": "2024-01-02",
                "currency_pair": "EUR_JPY",
                "prediction": 0.02,
                "actual_return": 0.015,
            }
        ])
    ],
) -> Annotated[
    List[RankedPredictionData],
    Check("algo_trade_dtype.checks.check_ranked_predictions"),
]:
    """Rank predictions across multiple currencies by date.

    Takes a list of prediction data and adds prediction_rank_pct
    (percentile rank within each date) in 0-1 scale.

    Args:
        predictions: List of prediction data for multiple currencies/dates

    Returns:
        List of ranked prediction data with prediction_rank_pct (0-1 scale)
    """
    if not predictions:
        return []

    df = pd.DataFrame(predictions)
    df["date"] = pd.to_datetime(df["date"])

    df["prediction_rank_pct"] = df.groupby("date")["prediction"].rank(pct=True)

    result: list[RankedPredictionData] = []
    for _, row in df.iterrows():
        result.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "currency_pair": row["currency_pair"],
                "prediction": float(row["prediction"]),
                "actual_return": float(row["actual_return"]),
                "prediction_rank_pct": float(row["prediction_rank_pct"]),
            }
        )

    return result


@transform
def select_top_currency(
    ranked_predictions: Annotated[
        List[RankedPredictionData],
        ExampleValue([
            {
                "date": "2024-01-01",
                "currency_pair": "USD_JPY",
                "prediction": 0.01,
                "actual_return": 0.005,
                "prediction_rank_pct": 0.5,
            },
            {
                "date": "2024-01-02",
                "currency_pair": "EUR_JPY",
                "prediction": 0.02,
                "actual_return": 0.015,
                "prediction_rank_pct": 1.0,
            }
        ])
    ],
    threshold_pct: float = 0.03,
) -> Annotated[
    List[SelectedCurrencyData],
    Check("algo_trade_dtype.checks.check_selected_currencies"),
]:
    """Select top and bottom currencies based on prediction ranking.

    Selects currencies with rank_pct >= (1 - threshold_pct) for BUY signals
    and rank_pct <= threshold_pct for SELL signals.

    Args:
        ranked_predictions: List of ranked prediction data
        threshold_pct: Threshold for top/bottom selection (default 0.03 = 3%)

    Returns:
        List of selected currencies with signals
    """
    if not ranked_predictions:
        return []

    if not 0 < threshold_pct <= _MAX_THRESHOLD_PCT:
        msg = f"threshold_pct must be in (0, {_MAX_THRESHOLD_PCT}], got {threshold_pct}"
        raise ValueError(msg)

    result: list[SelectedCurrencyData] = []

    for item in ranked_predictions:
        rank_pct = item["prediction_rank_pct"]

        if rank_pct >= (1.0 - threshold_pct):
            signal = 1.0
        elif rank_pct <= threshold_pct:
            signal = -1.0
        else:
            continue

        result.append(
            {
                "date": item["date"],
                "currency_pair": item["currency_pair"],
                "prediction": item["prediction"],
                "actual_return": item["actual_return"],
                "prediction_rank_pct": item["prediction_rank_pct"],
                "signal": signal,
            }
        )

    return result


@transform
def simulate_buy_scenario(
    selected_currencies: Annotated[
        List[SelectedCurrencyData],
        ExampleValue([
            {
                "date": "2024-01-01",
                "currency_pair": "EUR_JPY",
                "prediction": 0.02,
                "actual_return": 0.015,
                "prediction_rank_pct": 1.0,
                "signal": 1.0,
            },
            {
                "date": "2024-01-02",
                "currency_pair": "GBP_JPY",
                "prediction": -0.01,
                "actual_return": -0.005,
                "prediction_rank_pct": 0.0,
                "signal": -1.0,
            }
        ])
    ],
) -> Annotated[
    SimulationResult,
    Check("algo_trade_dtype.checks.check_simulation_result"),
]:
    """Simulate trading scenario based on selected currencies.

    Calculates portfolio returns using equal-weight allocation across
    selected currencies with their signals.

    Args:
        selected_currencies: List of selected currency data with signals

    Returns:
        SimulationResult with keys: date, portfolio_return, n_positions
    """
    if not selected_currencies:
        return {
            "date": [],
            "portfolio_return": [],
            "n_positions": [],
        }

    df = pd.DataFrame(selected_currencies)
    df["date"] = pd.to_datetime(df["date"])

    df["position_return"] = df["signal"] * df["actual_return"]

    daily_returns = (
        df.groupby("date")
        .agg(
            portfolio_return=("position_return", "mean"),
            n_positions=("signal", lambda x: (x != 0).sum()),
        )
        .reset_index()
    )

    daily_returns["portfolio_return"] = daily_returns["portfolio_return"].fillna(0.0)

    return {
        "date": [d.strftime("%Y-%m-%d") for d in daily_returns["date"]],
        "portfolio_return": daily_returns["portfolio_return"].tolist(),
        "n_positions": daily_returns["n_positions"].astype(int).tolist(),
    }


@transform
def calculate_performance_metrics(
    simulation_result: SimulationResult,
    annual_periods: int = 252,
) -> PerformanceMetrics:
    """Calculate portfolio performance metrics from simulation results.

    Args:
        simulation_result: SimulationResult with keys: date, portfolio_return,
            n_positions
        annual_periods: Number of periods per year for annualization (default 252)

    Returns:
        PerformanceMetrics with annual_return, annual_volatility, sharpe_ratio,
        max_drawdown, and calmar_ratio
    """
    if not simulation_result or not simulation_result.get("portfolio_return"):
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    returns = np.array(simulation_result["portfolio_return"])

    if len(returns) == 0:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    annual_return = float(np.mean(returns) * annual_periods)
    annual_volatility = float(np.std(returns, ddof=1) * np.sqrt(annual_periods))

    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(np.min(drawdown))

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }


__all__ = [
    "rank_predictions",
    "select_top_currency",
    "simulate_buy_scenario",
    "calculate_performance_metrics",
]
