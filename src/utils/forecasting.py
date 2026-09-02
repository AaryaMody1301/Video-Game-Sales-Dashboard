"""Forecasting helpers with time-aware validation and explicit model behavior."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima.model import ARIMA


SUPPORTED_MODELS = {"linear", "poly", "ridge", "arima"}


class ForecastError(RuntimeError):
    """Raised when a requested forecast model cannot be fit reliably."""


def _prepare_series(years, sales) -> Tuple[np.ndarray, np.ndarray]:
    years_array = np.asarray(years, dtype=float).reshape(-1)
    sales_array = np.asarray(sales, dtype=float).reshape(-1)

    if len(years_array) != len(sales_array):
        raise ValueError("Years and sales must contain the same number of values.")

    valid = np.isfinite(years_array) & np.isfinite(sales_array)
    years_array = years_array[valid]
    sales_array = sales_array[valid]

    if len(years_array) < 3:
        raise ForecastError("At least three historical observations are required.")

    order = np.argsort(years_array)
    return years_array[order], sales_array[order]


def chronological_split(years, sales, test_fraction: float = 0.2):
    """Split a time series without shuffling, reserving the latest years for validation."""
    years_array, sales_array = _prepare_series(years, sales)

    if len(years_array) < 5:
        return years_array, np.array([]), sales_array, np.array([])

    test_size = max(1, int(np.ceil(len(years_array) * test_fraction)))
    test_size = min(test_size, len(years_array) - 3)
    split_index = len(years_array) - test_size

    return (
        years_array[:split_index],
        years_array[split_index:],
        sales_array[:split_index],
        sales_array[split_index:],
    )


def _build_regression_model(model_type: str):
    if model_type == "linear":
        return LinearRegression()
    if model_type == "poly":
        return make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    if model_type == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(f"Unsupported regression model: {model_type}")


def _time_series_cv_rmse(model, years: np.ndarray, sales: np.ndarray):
    if len(years) < 8:
        return None

    n_splits = min(4, len(years) - 2)
    if n_splits < 2:
        return None

    try:
        splitter = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(
            model,
            years.reshape(-1, 1),
            sales,
            cv=splitter,
            scoring="neg_root_mean_squared_error",
        )
        return float(-np.mean(scores))
    except ValueError:
        return None


def _regression_forecast(model_type, years, sales, future_years):
    model = _build_regression_model(model_type)
    x = years.reshape(-1, 1)
    future_x = np.asarray(future_years, dtype=float).reshape(-1, 1)

    model.fit(x, sales)
    fitted = model.predict(x)
    raw_predictions = model.predict(future_x)

    rmse = float(np.sqrt(mean_squared_error(sales, fitted)))
    r2 = float(r2_score(sales, fitted)) if len(sales) > 1 else None
    cv_rmse = _time_series_cv_rmse(_build_regression_model(model_type), years, sales)

    predictions = np.maximum(raw_predictions, 0.0)
    interval_width = 1.96 * rmse
    lower = np.maximum(predictions - interval_width, 0.0)
    upper = np.maximum(predictions + interval_width, lower)

    return {
        "predictions": predictions,
        "lower": lower,
        "upper": upper,
        "stats": {
            "model": model_type,
            "rmse": rmse,
            "r2": r2,
            "cv_rmse": cv_rmse,
            "interval_label": "Approx. 95% interval",
        },
    }


def _arima_forecast(years, sales, future_years):
    steps = len(np.asarray(future_years).reshape(-1))
    if steps < 1:
        raise ValueError("At least one future year is required.")

    attempts = [(1, 1, 0), (0, 1, 0)]
    last_error = None
    results = None
    fitted_order = None

    for order in attempts:
        try:
            results = ARIMA(sales, order=order).fit()
            fitted_order = order
            break
        except Exception as exc:  # statsmodels raises several fit-time exception types
            last_error = exc

    if results is None:
        raise ForecastError(f"ARIMA fitting failed: {last_error}")

    forecast = results.get_forecast(steps=steps)
    predictions = np.maximum(np.asarray(forecast.predicted_mean, dtype=float), 0.0)
    confidence = np.asarray(forecast.conf_int(alpha=0.05), dtype=float)
    lower = np.maximum(confidence[:, 0], 0.0)
    upper = np.maximum(confidence[:, 1], lower)

    residuals = np.asarray(results.resid, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    rmse = float(np.sqrt(np.mean(np.square(residuals)))) if residuals.size else None

    return {
        "predictions": predictions,
        "lower": lower,
        "upper": upper,
        "stats": {
            "model": "arima",
            "rmse": rmse,
            "r2": None,
            "cv_rmse": None,
            "order": fitted_order,
            "interval_label": "95% forecast interval",
        },
    }


def fit_forecast_model(model_type, years, sales, future_years) -> Dict[str, object]:
    """Fit the requested model and forecast future years without substituting model families."""
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type: {model_type}")

    years_array, sales_array = _prepare_series(years, sales)
    future_years_array = np.asarray(future_years, dtype=float).reshape(-1)

    if model_type == "arima":
        return _arima_forecast(years_array, sales_array, future_years_array)

    return _regression_forecast(
        model_type,
        years_array,
        sales_array,
        future_years_array,
    )


def evaluate_model_chronologically(model_type, years, sales) -> Dict[str, float]:
    """Evaluate on the latest observations so validation respects time ordering."""
    train_years, test_years, train_sales, test_sales = chronological_split(years, sales)

    if len(test_years) < 2:
        return {}

    result = fit_forecast_model(model_type, train_years, train_sales, test_years)
    predictions = np.asarray(result["predictions"], dtype=float)

    return {
        "validation_rmse": float(np.sqrt(mean_squared_error(test_sales, predictions))),
        "validation_r2": float(r2_score(test_sales, predictions)),
        "validation_start_year": float(test_years[0]),
        "validation_end_year": float(test_years[-1]),
    }
