"""Training Transform functions for algorithmic trading pipeline."""

from __future__ import annotations

from typing import Annotated, Any, Mapping

import numpy as np
import pandas as pd

from xform_core import Check, transform

from algo_trade_dtypes.types import (
    FoldResult,
    CVResult,
    TimeSeriesSplitConfig,
)


def convert_nullable_dtypes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert nullable pandas dtypes to standard numpy dtypes for compatibility.

    This function converts nullable dtypes (Float64, Int64, boolean) to standard
    numpy dtypes (float64, int64, bool) to ensure compatibility with libraries
    like LightGBM that may not support nullable dtypes.
    """
    if df.empty:
        return df

    dtype_mapping = {"Float64": "float64", "Int64": "int64", "boolean": "bool"}

    converted_df = df.copy()
    for col in converted_df.columns:
        dtype_str = str(converted_df[col].dtype)
        if dtype_str in dtype_mapping:
            converted_df[col] = converted_df[col].astype(dtype_mapping[dtype_str])

    return converted_df


@transform
def get_cv_splits(
    n_samples: int,
    config: TimeSeriesSplitConfig | None = None,
) -> Annotated[
    list[tuple[list[int], list[int]]], Check("algo_trade_dtypes.checks.check_cv_splits")
]:
    """Generate time-series CV splits using sklearn TimeSeriesSplit."""
    if config is None:
        config = {
            "n_splits": 5,
            "test_size": None,
            "gap": 0,
        }

    n_splits = config.get("n_splits", 5)
    test_size = config.get("test_size", None)
    gap = config.get("gap", 0)

    return _time_series_split(n_samples, n_splits, test_size, gap)


def _time_series_split(
    n_samples: int,
    n_splits: int,
    test_size: int | None,
    gap: int,
) -> list[tuple[list[int], list[int]]]:
    """Standard time series split implementation."""
    splits: list[tuple[list[int], list[int]]] = []

    if test_size is None:
        # Auto-calculate test_size
        test_size = n_samples // (n_splits + 1)

    for i in range(n_splits):
        test_start = (i + 1) * test_size
        test_end = test_start + test_size

        test_end = min(test_end, n_samples)

        train_end = test_start - gap
        if train_end <= 0:
            continue

        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, test_end))

        if len(train_indices) > 0 and len(test_indices) > 0:
            splits.append((train_indices, test_indices))

    return splits


def _expanding_window_split(
    n_samples: int,
    n_splits: int,
    test_size: int | None,
    gap: int,
) -> list[tuple[list[int], list[int]]]:
    """Expanding window split implementation."""
    splits: list[tuple[list[int], list[int]]] = []

    if test_size is None:
        test_size = n_samples // (n_splits + 1)

    for i in range(n_splits):
        test_start = n_samples - test_size * (n_splits - i)
        test_end = test_start + test_size

        train_end = test_start - gap
        if train_end <= 0:
            continue

        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, min(test_end, n_samples)))

        if len(train_indices) > 0 and len(test_indices) > 0:
            splits.append((train_indices, test_indices))

    return splits


def _sliding_window_split(
    n_samples: int,
    n_splits: int,
    test_size: int | None,
    gap: int,
) -> list[tuple[list[int], list[int]]]:
    """Sliding window split implementation."""
    splits: list[tuple[list[int], list[int]]] = []

    if test_size is None:
        test_size = n_samples // (n_splits + 1)

    train_size = n_samples // 2  # Use half of data for training

    for i in range(n_splits):
        test_start = n_samples - test_size * (n_splits - i)
        test_end = test_start + test_size

        train_start = max(0, test_start - gap - train_size)
        train_end = test_start - gap

        if train_end <= train_start:
            continue

        train_indices = list(range(train_start, train_end))
        test_indices = list(range(test_start, min(test_end, n_samples)))

        if len(train_indices) > 0 and len(test_indices) > 0:
            splits.append((train_indices, test_indices))

    return splits


def train_single_fold(
    features: pd.DataFrame,
    target: pd.DataFrame,
    train_indices: list[int],
    valid_indices: list[int],
    fold_id: int,
    lgbm_params: Mapping[str, Any] | None = None,
) -> FoldResult:
    """Train model on a single fold and return results.

    Helper function for ML training - not a @transform function.
    Uses LightGBM for regression.
    """
    import lightgbm as lgb

    if lgbm_params is None:
        params: dict[str, Any] = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "random_state": 42,
        }
    else:
        params = dict(lgbm_params)

    # Convert nullable dtypes for LightGBM compatibility
    features_converted = convert_nullable_dtypes(features)
    target_converted = convert_nullable_dtypes(target)

    X_train = features_converted.iloc[train_indices]
    y_train = target_converted.iloc[train_indices]["target"]
    X_valid = features_converted.iloc[valid_indices]
    y_valid = target_converted.iloc[valid_indices]["target"]

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    train_score = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))
    valid_score = float(np.sqrt(np.mean((y_valid - valid_pred) ** 2)))

    feature_importance = dict(
        zip(features.columns, model.feature_importances_, strict=False)
    )

    fold_result: FoldResult = {
        "fold_id": fold_id,
        "train_indices": train_indices,
        "valid_indices": valid_indices,
        "train_score": train_score,
        "valid_score": valid_score,
        "predictions": valid_pred.tolist(),
        "feature_importance": feature_importance,
    }
    return fold_result


def aggregate_cv_results(
    fold_results: list[FoldResult],
    actuals: pd.Series,
) -> CVResult:
    """Aggregate results from all CV folds and collect OOS actuals.

    Helper function for result aggregation - not a @transform function.

    Parameters:
        actuals: 全体の実測値（OOS インデックスから抽出するため）
    """
    if not fold_results:
        return {
            "fold_results": [],
            "mean_score": 0.0,
            "std_score": 0.0,
            "oos_predictions": [],
        }

    oos_predictions: list[float] = []
    oos_indices: list[int] = []

    for fold in fold_results:
        # Append predictions for this fold
        oos_predictions.extend(fold["predictions"])
        # Append validation indices for this fold
        oos_indices.extend(fold["valid_indices"])

    # Calculate mean and std scores
    valid_scores = [fold["valid_score"] for fold in fold_results]
    mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
    std_score = float(np.std(valid_scores)) if valid_scores else 0.0

    return {
        "fold_results": fold_results,
        "mean_score": mean_score,
        "std_score": std_score,
        "oos_predictions": oos_predictions,
    }


def extract_predictions(
    cv_result: CVResult,
    dates: list[str],
    currency_pairs: list[str],
    actual_returns: pd.Series,
) -> list[dict[str, object]]:
    """Extract predictions from CV result and format as PredictionData.

    Helper function to convert CV predictions to PredictionData format.
    Not a @transform function - technical data reshaping.
    """
    oos_predictions = cv_result["oos_predictions"]

    if len(oos_predictions) != len(dates) or len(dates) != len(currency_pairs):
        raise ValueError(
            f"Length mismatch: predictions={len(oos_predictions)}, "
            f"dates={len(dates)}, currency_pairs={len(currency_pairs)}"
        )

    result: list[dict[str, object]] = []
    for _i, (pred, date, pair, actual) in enumerate(
        zip(oos_predictions, dates, currency_pairs, actual_returns, strict=False)
    ):
        result.append(
            {
                "date": date,
                "currency_pair": pair,
                "prediction": float(pred),
                "actual_return": float(actual),
            }
        )

    return result


@transform
def calculate_rmse(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Annotated[float, Check("algo_trade_dtypes.checks.check_nonnegative_float")]:
    """Calculate Root Mean Squared Error (RMSE) between true and predicted values."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
        return 0.0

    if len(y_true_arr) != len(y_pred_arr):
        msg = (
            f"y_true and y_pred must have same length: "
            f"{len(y_true_arr)} != {len(y_pred_arr)}"
        )
        raise ValueError(msg)

    mse = np.mean((y_true_arr - y_pred_arr) ** 2)
    rmse = np.sqrt(mse)

    return float(rmse)


__all__ = [
    "convert_nullable_dtypes",
    "get_cv_splits",
    "calculate_rmse",
    "train_single_fold",
    "aggregate_cv_results",
    "extract_predictions",
]
