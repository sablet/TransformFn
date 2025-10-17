"""Simulation and evaluation Transform functions for algorithmic trading pipeline."""

from __future__ import annotations

from typing import Annotated, List, Literal

from xform_core import Check, ExampleValue

import numpy as np
import pandas as pd

from xform_core import transform

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
        ExampleValue(
            [
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
                },
            ]
        ),
    ],
    *,
    method: Literal["percentile", "ordinal", "zscore", "minmax"] = "percentile",
    groupby: Literal["date", "currency_pair", "none"] = "date",
) -> Annotated[
    List[RankedPredictionData],
    Check("algo_trade_dtype.checks.check_ranked_predictions"),
]:
    """Rank predictions with configurable method and grouping.

    Args:
        predictions: List of prediction data for multiple currencies/dates
        method: Ranking method - "percentile" (0-1 scale), "ordinal" (1-N),
                "zscore" (standardized), "minmax" (0-1 normalized)
        groupby: Grouping strategy - "date" (within each date),
                 "currency_pair" (within each currency), "none" (all at once)

    Returns:
        List of ranked prediction data with prediction_rank_pct
    """
    if not predictions:
        return []

    df = pd.DataFrame(predictions)
    df["date"] = pd.to_datetime(df["date"])

    # Apply ranking based on method and grouping
    if groupby == "none":
        df = _apply_ranking_method(df, "prediction", method)
    elif groupby == "date":
        df = (
            df.groupby("date", group_keys=False)[df.columns.tolist()]
            .apply(
                lambda x: _apply_ranking_method(x, "prediction", method),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
    elif groupby == "currency_pair":
        df = (
            df.groupby("currency_pair", group_keys=False)[df.columns.tolist()]
            .apply(
                lambda x: _apply_ranking_method(x, "prediction", method),
                include_groups=False,
            )
            .reset_index(drop=True)
        )

    result: list[RankedPredictionData] = []
    for _, row in df.iterrows():
        result.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "currency_pair": row["currency_pair"],
                "prediction": float(row["prediction"]),
                "actual_return": float(row["actual_return"]),
                "prediction_rank_pct": float(row["prediction_rank"]),
            }
        )

    return result


def _apply_ranking_method(df: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
    """Apply the specified ranking method to a DataFrame column."""
    df = df.copy()
    values = df[column]

    if method == "percentile":
        # Use pandas rank with pct=True for percentile ranking
        df["prediction_rank"] = values.rank(pct=True, method="average")
    elif method == "ordinal":
        # Use pandas rank with ordinal method for 1-N ranking
        df["prediction_rank"] = values.rank(method="min").astype(int)
    elif method == "zscore":
        # Calculate z-scores
        mean_val = values.mean()
        std_val = values.std()
        if std_val == 0:
            df["prediction_rank"] = 0.0
        else:
            df["prediction_rank"] = (values - mean_val) / std_val
    elif method == "minmax":
        # Normalize to 0-1 scale
        min_val = values.min()
        max_val = values.max()
        if min_val == max_val:
            df["prediction_rank"] = 0.0
        else:
            df["prediction_rank"] = (values - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unsupported ranking method: {method}")

    return df


@transform
def select_top_currency(
    ranked_predictions: Annotated[
        List[RankedPredictionData],
        ExampleValue(
            [
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
                },
            ]
        ),
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
        ExampleValue(
            [
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
                },
            ]
        ),
    ],
    *,
    allocation_method: Literal[
        "equal", "prediction_weighted", "rank_weighted"
    ] = "equal",
    rebalance_freq: Literal["D", "W", "M"] = "D",
) -> Annotated[
    SimulationResult,
    Check("algo_trade_dtype.checks.check_simulation_result"),
]:
    """Simulate trading scenario with configurable allocation and rebalancing.

    Args:
        selected_currencies: List of selected currency data with signals
        allocation_method: How to weight positions - "equal" (equal weight),
                           "prediction_weighted" (weight by absolute prediction),
                           "rank_weighted" (weight by prediction rank)
        rebalance_freq: How often to rebalance - "D" (daily), "W" (weekly),
                        "M" (monthly)

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

    # Apply rebalancing frequency by resampling to the specified frequency
    if rebalance_freq != "D":
        df = _apply_rebalancing_frequency(df, rebalance_freq)

    # Apply allocation method to calculate position weights
    df = _apply_allocation_method(df, allocation_method)

    # Calculate position returns based on weights
    df["position_return"] = df["weight"] * df["signal"] * df["actual_return"]

    # Group by date and aggregate portfolio returns
    daily_returns = (
        df.groupby("date")
        .agg(
            portfolio_return=(
                "position_return",
                "sum",
            ),  # Sum instead of mean for weighted returns
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


def _apply_rebalancing_frequency(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Apply rebalancing frequency by resampling data."""
    if freq == "W":
        # Group by week
        df["date"] = df["date"].dt.to_period("W").dt.start_time
    elif freq == "M":
        # Group by month
        df["date"] = df["date"].dt.to_period("M").dt.start_time
    # For "D" (daily), no changes needed

    return df


def _apply_allocation_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Apply the specified allocation method to calculate position weights."""
    df = df.copy()

    if method == "equal":
        # Equal weight allocation - each position gets 1 / n_positions weight
        df["weight"] = 1.0 / df.groupby("date")["signal"].transform("size")
    elif method == "prediction_weighted":
        # Weight by absolute value of prediction
        df["abs_prediction"] = df["prediction"].abs()
        total_abs_predictions = df.groupby("date")["abs_prediction"].transform("sum")
        df["weight"] = df["abs_prediction"] / total_abs_predictions
        df = df.drop(columns=["abs_prediction"])  # Clean up temporary column
    elif method == "rank_weighted":
        # Weight by rank of prediction
        df["rank"] = df.groupby("date")["prediction"].rank(method="average")
        total_ranks = df.groupby("date")["rank"].transform("sum")
        df["weight"] = df["rank"] / total_ranks
        df = df.drop(columns=["rank"])  # Clean up temporary column
    else:
        raise ValueError(f"Unsupported allocation method: {method}")

    # Fill any missing weights with 0
    df["weight"] = df["weight"].fillna(0.0)

    return df


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
