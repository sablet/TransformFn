"""Tests for simulation Transform functions."""

from __future__ import annotations

import pytest

from algo_trade_app.simulation import (
    calculate_performance_metrics,
    rank_predictions,
    select_top_currency,
    simulate_buy_scenario,
)


class TestRankPredictions:
    """Tests for rank_predictions Transform."""

    def test_rank_predictions_basic(self) -> None:
        """Test basic ranking functionality."""
        predictions = [
            {
                "date": "2024-01-01",
                "currency_pair": "USD_JPY",
                "prediction": 0.5,
                "actual_return": 0.01,
            },
            {
                "date": "2024-01-01",
                "currency_pair": "EUR_JPY",
                "prediction": 0.8,
                "actual_return": 0.02,
            },
            {
                "date": "2024-01-01",
                "currency_pair": "GBP_JPY",
                "prediction": 0.2,
                "actual_return": -0.01,
            },
        ]

        result = rank_predictions(predictions)

        assert len(result) == 3
        assert all("prediction_rank_pct" in item for item in result)

        eur_item = next(item for item in result if item["currency_pair"] == "EUR_JPY")
        gbp_item = next(item for item in result if item["currency_pair"] == "GBP_JPY")
        usd_item = next(item for item in result if item["currency_pair"] == "USD_JPY")

        # Check ranking: EUR_JPY (0.8) should be 1.0, USD_JPY (0.5) should be ~0.67, GBP_JPY (0.2) should be ~0.33
        assert eur_item["prediction_rank_pct"] == 1.0
        assert gbp_item["prediction_rank_pct"] < usd_item["prediction_rank_pct"]
        assert usd_item["prediction_rank_pct"] < eur_item["prediction_rank_pct"]

    def test_rank_predictions_empty(self) -> None:
        """Test with empty list."""
        result = rank_predictions([])
        assert result == []

    def test_rank_predictions_multiple_dates(self) -> None:
        """Test ranking across multiple dates."""
        predictions = [
            {
                "date": "2024-01-01",
                "currency_pair": "USD_JPY",
                "prediction": 0.5,
                "actual_return": 0.01,
            },
            {
                "date": "2024-01-01",
                "currency_pair": "EUR_JPY",
                "prediction": 0.8,
                "actual_return": 0.02,
            },
            {
                "date": "2024-01-02",
                "currency_pair": "USD_JPY",
                "prediction": 0.2,
                "actual_return": -0.01,
            },
            {
                "date": "2024-01-02",
                "currency_pair": "EUR_JPY",
                "prediction": 0.9,
                "actual_return": 0.03,
            },
        ]

        result = rank_predictions(predictions)

        assert len(result) == 4

        # Check that ranking is done within each date
        date1_items = [item for item in result if item["date"] == "2024-01-01"]
        date2_items = [item for item in result if item["date"] == "2024-01-02"]

        assert len(date1_items) == 2
        assert len(date2_items) == 2


class TestSelectTopCurrency:
    """Tests for select_top_currency Transform."""

    def test_select_top_currency_basic(self) -> None:
        """Test basic currency selection with 0-1 scale."""
        ranked_predictions = [
            {
                "date": "2024-01-01",
                "currency_pair": "PAIR_0",
                "prediction": 0.1,
                "actual_return": 0.01,
                "prediction_rank_pct": 0.1,
            },
            {
                "date": "2024-01-01",
                "currency_pair": "PAIR_1",
                "prediction": 0.2,
                "actual_return": 0.02,
                "prediction_rank_pct": 0.2,
            },
            {
                "date": "2024-01-01",
                "currency_pair": "PAIR_2",
                "prediction": 0.5,
                "actual_return": 0.03,
                "prediction_rank_pct": 0.5,
            },
            {
                "date": "2024-01-01",
                "currency_pair": "PAIR_3",
                "prediction": 0.8,
                "actual_return": 0.04,
                "prediction_rank_pct": 0.8,
            },
            {
                "date": "2024-01-01",
                "currency_pair": "PAIR_4",
                "prediction": 0.9,
                "actual_return": 0.05,
                "prediction_rank_pct": 0.9,
            },
        ]

        result = select_top_currency(ranked_predictions, threshold_pct=0.2)

        assert len(result) >= 2  # At least top and bottom
        assert all("signal" in item for item in result)
        assert all(item["signal"] in [-1.0, 1.0] for item in result)

        # Verify top currencies have signal=1.0
        top_items = [item for item in result if item["prediction_rank_pct"] >= 0.8]
        assert all(item["signal"] == 1.0 for item in top_items)

        # Verify bottom currencies have signal=-1.0
        bottom_items = [item for item in result if item["prediction_rank_pct"] <= 0.2]
        assert all(item["signal"] == -1.0 for item in bottom_items)

    def test_select_top_currency_default_threshold(self) -> None:
        """Test with default threshold (0.03)."""
        ranked_predictions = [
            {
                "date": "2024-01-01",
                "currency_pair": "PAIR_0",
                "prediction": 0.1,
                "actual_return": 0.01,
                "prediction_rank_pct": 0.01,
            },
            {
                "date": "2024-01-01",
                "currency_pair": "PAIR_1",
                "prediction": 0.5,
                "actual_return": 0.02,
                "prediction_rank_pct": 0.5,
            },
            {
                "date": "2024-01-01",
                "currency_pair": "PAIR_2",
                "prediction": 0.9,
                "actual_return": 0.03,
                "prediction_rank_pct": 0.99,
            },
        ]

        result = select_top_currency(ranked_predictions)

        # With threshold 0.03: rank_pct <= 0.03 for SELL, >= 0.97 for BUY
        assert len(result) == 2
        assert any(item["signal"] == -1.0 for item in result)
        assert any(item["signal"] == 1.0 for item in result)

    def test_select_top_currency_empty(self) -> None:
        """Test with empty list."""
        result = select_top_currency([])
        assert result == []

    def test_select_top_currency_no_selection(self) -> None:
        """Test when no currencies meet threshold."""
        ranked_predictions = [
            {
                "date": "2024-01-01",
                "currency_pair": "PAIR_0",
                "prediction": 0.5,
                "actual_return": 0.01,
                "prediction_rank_pct": 0.5,
            },
        ]

        result = select_top_currency(ranked_predictions, threshold_pct=0.1)
        assert result == []

    def test_select_top_currency_invalid_threshold(self) -> None:
        """Test error handling for invalid threshold."""
        ranked_predictions = [
            {
                "date": "2024-01-01",
                "currency_pair": "USD_JPY",
                "prediction": 0.5,
                "actual_return": 0.01,
                "prediction_rank_pct": 0.5,
            },
        ]

        with pytest.raises(ValueError, match="threshold_pct must be in"):
            select_top_currency(ranked_predictions, threshold_pct=0.6)

        with pytest.raises(ValueError, match="threshold_pct must be in"):
            select_top_currency(ranked_predictions, threshold_pct=0.0)


class TestSimulateBuyScenario:
    """Tests for simulate_buy_scenario Transform."""

    def test_simulate_buy_scenario_basic(self) -> None:
        """Test basic simulation."""
        selected_currencies = [
            {
                "date": "2024-01-01",
                "currency_pair": "USD_JPY",
                "prediction": 0.5,
                "actual_return": 0.01,
                "prediction_rank_pct": 0.9,
                "signal": 1.0,
            },
            {
                "date": "2024-01-01",
                "currency_pair": "EUR_JPY",
                "prediction": -0.2,
                "actual_return": -0.02,
                "prediction_rank_pct": 0.1,
                "signal": -1.0,
            },
            {
                "date": "2024-01-02",
                "currency_pair": "USD_JPY",
                "prediction": 0.6,
                "actual_return": 0.03,
                "prediction_rank_pct": 0.95,
                "signal": 1.0,
            },
        ]

        result = simulate_buy_scenario(selected_currencies)

        assert "date" in result
        assert "portfolio_return" in result
        assert "n_positions" in result

        assert len(result["date"]) == 2  # 2 unique dates
        assert len(result["portfolio_return"]) == 2
        assert len(result["n_positions"]) == 2

        # Check that returns are calculated correctly
        # Date 1: (1.0 * 0.01 + (-1.0) * (-0.02)) / 2 = (0.01 + 0.02) / 2 = 0.015
        assert abs(result["portfolio_return"][0] - 0.015) < 1e-6

        # Check n_positions
        assert result["n_positions"][0] == 2  # Date 1 has 2 positions
        assert result["n_positions"][1] == 1  # Date 2 has 1 position

    def test_simulate_buy_scenario_empty(self) -> None:
        """Test with empty list."""
        result = simulate_buy_scenario([])

        assert result["date"] == []
        assert result["portfolio_return"] == []
        assert result["n_positions"] == []

    def test_simulate_buy_scenario_single_position(self) -> None:
        """Test with single position per date."""
        selected_currencies = [
            {
                "date": "2024-01-01",
                "currency_pair": "USD_JPY",
                "prediction": 0.5,
                "actual_return": 0.02,
                "prediction_rank_pct": 0.9,
                "signal": 1.0,
            },
        ]

        result = simulate_buy_scenario(selected_currencies)

        assert len(result["date"]) == 1
        assert result["portfolio_return"][0] == 0.02
        assert result["n_positions"][0] == 1


class TestCalculatePerformanceMetrics:
    """Tests for calculate_performance_metrics Transform."""

    def test_calculate_performance_metrics_basic(self) -> None:
        """Test basic metrics calculation."""
        simulation_result = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "portfolio_return": [0.01, -0.005, 0.015],
            "n_positions": [2, 2, 3],
        }

        result = calculate_performance_metrics(simulation_result)

        assert isinstance(result, dict)
        assert "annual_return" in result
        assert "annual_volatility" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "calmar_ratio" in result

        # Check all values are finite
        for key, value in result.items():
            assert isinstance(value, float)
            assert not isinstance(value, bool)
            import math

            assert math.isfinite(value), f"{key} is not finite: {value}"

    def test_calculate_performance_metrics_empty(self) -> None:
        """Test with empty result."""
        simulation_result = {
            "date": [],
            "portfolio_return": [],
            "n_positions": [],
        }

        result = calculate_performance_metrics(simulation_result)

        assert result["annual_return"] == 0.0
        assert result["annual_volatility"] == 0.0
        assert result["sharpe_ratio"] == 0.0
        assert result["max_drawdown"] == 0.0
        assert result["calmar_ratio"] == 0.0

    def test_calculate_performance_metrics_custom_periods(self) -> None:
        """Test with custom annual periods."""
        simulation_result = {
            "date": ["2024-01-01", "2024-01-02"],
            "portfolio_return": [0.01, 0.02],
            "n_positions": [2, 2],
        }

        result_252 = calculate_performance_metrics(
            simulation_result, annual_periods=252
        )
        result_365 = calculate_performance_metrics(
            simulation_result, annual_periods=365
        )

        # Annual return should scale with annual_periods
        assert result_365["annual_return"] > result_252["annual_return"]
        assert result_365["annual_volatility"] > result_252["annual_volatility"]

    def test_calculate_performance_metrics_all_positive_returns(self) -> None:
        """Test with all positive returns."""
        simulation_result = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "portfolio_return": [0.01, 0.02, 0.015],
            "n_positions": [2, 2, 2],
        }

        result = calculate_performance_metrics(simulation_result)

        assert result["annual_return"] > 0
        assert result["annual_volatility"] > 0
        assert result["sharpe_ratio"] > 0
        assert result["max_drawdown"] == 0.0  # No drawdown with all positive returns

    def test_calculate_performance_metrics_with_drawdown(self) -> None:
        """Test metrics calculation with drawdown."""
        simulation_result = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "portfolio_return": [0.1, -0.05, -0.03],
            "n_positions": [2, 2, 2],
        }

        result = calculate_performance_metrics(simulation_result)

        assert result["max_drawdown"] < 0
        assert abs(result["calmar_ratio"]) > 0
