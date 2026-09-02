import numpy as np

from src.utils.forecasting import (
    chronological_split,
    evaluate_model_chronologically,
    fit_forecast_model,
)


def test_chronological_split_reserves_latest_years():
    years = np.arange(2010, 2020)
    sales = np.arange(10, dtype=float)

    train_years, test_years, train_sales, test_sales = chronological_split(years, sales)

    assert train_years.tolist() == list(range(2010, 2018))
    assert test_years.tolist() == [2018, 2019]
    assert train_sales.tolist() == list(np.arange(8, dtype=float))
    assert test_sales.tolist() == [8.0, 9.0]


def test_regression_forecast_uses_requested_model_and_clips_negative_sales():
    years = np.arange(2010, 2020)
    declining_sales = np.linspace(9.0, 0.0, len(years))

    result = fit_forecast_model("linear", years, declining_sales, [2020, 2021, 2022])

    assert result["stats"]["model"] == "linear"
    assert len(result["predictions"]) == 3
    assert np.all(result["predictions"] >= 0)
    assert np.all(result["lower"] >= 0)


def test_polynomial_model_reports_time_series_cross_validation():
    years = np.arange(2005, 2020)
    sales = 2.0 + 0.1 * (years - 2005) ** 2

    result = fit_forecast_model("poly", years, sales, [2020, 2021])

    assert result["stats"]["model"] == "poly"
    assert result["stats"]["cv_rmse"] is not None


def test_arima_uses_current_statsmodels_fit_api_without_model_substitution():
    years = np.arange(2000, 2014)
    sales = np.array([4.0, 4.4, 4.1, 4.8, 5.0, 5.3, 5.1, 5.8, 6.0, 6.4, 6.2, 6.8, 7.0, 7.4])

    result = fit_forecast_model("arima", years, sales, [2014, 2015, 2016])

    assert result["stats"]["model"] == "arima"
    assert result["stats"]["order"] in {(1, 1, 0), (0, 1, 0)}
    assert len(result["predictions"]) == 3
    assert np.all(result["predictions"] >= 0)


def test_validation_metrics_cover_latest_holdout_period():
    years = np.arange(2000, 2012)
    sales = np.linspace(2.0, 8.0, len(years))

    validation = evaluate_model_chronologically("ridge", years, sales)

    assert validation["validation_start_year"] >= 2009
    assert validation["validation_end_year"] == 2011
    assert validation["validation_rmse"] >= 0
