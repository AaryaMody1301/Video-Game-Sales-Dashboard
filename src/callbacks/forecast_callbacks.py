"""Callbacks for forecasting functionality."""

from functools import lru_cache
import logging
import time

from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data.data_loader import apply_filters
from src.utils.forecasting import (
    ForecastError,
    evaluate_model_chronologically,
    fit_forecast_model,
)


logger = logging.getLogger(__name__)
MODEL_CACHE_SIZE = 32
MODEL_LABELS = {
    "linear": "Linear Regression",
    "poly": "Polynomial Regression",
    "ridge": "Ridge Regression",
    "arima": "ARIMA",
}


@lru_cache(maxsize=MODEL_CACHE_SIZE)
def fit_and_predict_cached(model_type, years_key, sales_key, future_years_key):
    """Cache deterministic forecasts for repeated filter/model combinations."""
    return fit_forecast_model(
        model_type,
        np.asarray(years_key, dtype=float),
        np.asarray(sales_key, dtype=float),
        np.asarray(future_years_key, dtype=float),
    )


def _error_figure(message):
    figure = px.line(title=message)
    figure.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
    )
    return figure


def _format_metrics(stats, validation_stats):
    details = []

    if stats.get("order") is not None:
        details.append(f"ARIMA order: {stats['order']}")
    if stats.get("rmse") is not None:
        details.append(f"Fit RMSE: {stats['rmse']:.2f}")
    if stats.get("cv_rmse") is not None:
        details.append(f"Time-series CV RMSE: {stats['cv_rmse']:.2f}")
    if validation_stats.get("validation_rmse") is not None:
        details.append(f"Latest-years RMSE: {validation_stats['validation_rmse']:.2f}")
    if validation_stats.get("validation_r2") is not None:
        details.append(f"Latest-years R²: {validation_stats['validation_r2']:.2f}")

    start_year = validation_stats.get("validation_start_year")
    end_year = validation_stats.get("validation_end_year")
    if start_year is not None and end_year is not None:
        details.append(f"Validation period: {int(start_year)}-{int(end_year)}")

    return details


def _build_sales_figure(yearly_sales, future_years, result, validation_stats, model_type):
    last_year = int(yearly_sales["release_year"].max())
    forecast_df = pd.DataFrame(
        {
            "release_year": future_years,
            "total_sales": result["predictions"],
            "data_type": "Forecast",
        }
    )
    historical_df = yearly_sales.copy()
    historical_df["data_type"] = "Historical"
    combined = pd.concat([historical_df, forecast_df], ignore_index=True)

    model_label = MODEL_LABELS.get(model_type, model_type)
    figure = px.line(
        combined,
        x="release_year",
        y="total_sales",
        color="data_type",
        title=f"Sales by Release Year Forecast ({model_label})",
        labels={"release_year": "Release Year", "total_sales": "Total Sales (millions)"},
        markers=True,
    )

    stats = result["stats"]
    interval_name = stats.get("interval_label", "Forecast interval")
    figure.add_trace(
        go.Scatter(
            x=future_years,
            y=result["upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=future_years,
            y=result["lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 100, 80, 0.2)",
            name=interval_name,
        )
    )

    metrics = _format_metrics(stats, validation_stats)
    if metrics:
        figure.add_annotation(
            x=0.03,
            y=0.97,
            xref="paper",
            yref="paper",
            text="<br>".join(metrics),
            showarrow=False,
            align="left",
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="gray",
            borderwidth=1,
        )

    figure.add_vline(x=last_year, line_dash="dot")
    return figure


def _build_genre_figure(filtered_df, future_years, model_type):
    genre_yearly = (
        filtered_df.dropna(subset=["release_year"])
        .groupby(["release_year", "genre"], observed=False)["total_sales"]
        .sum()
        .reset_index()
        .sort_values("release_year")
    )
    top_genres = (
        filtered_df.groupby("genre", observed=False)["total_sales"]
        .sum()
        .nlargest(5)
        .index.tolist()
    )

    historical_parts = []
    forecast_parts = []
    failed_genres = []

    for genre in top_genres:
        genre_data = genre_yearly[genre_yearly["genre"] == genre].dropna(subset=["total_sales"])
        if len(genre_data) < 5:
            continue

        historical = genre_data.copy()
        historical["data_type"] = "Historical"
        historical_parts.append(historical)

        years_key = tuple(map(float, genre_data["release_year"].to_numpy()))
        sales_key = tuple(map(float, genre_data["total_sales"].to_numpy()))
        future_key = tuple(map(float, future_years))

        try:
            result = fit_and_predict_cached(model_type, years_key, sales_key, future_key)
        except (ForecastError, ValueError) as exc:
            logger.warning("%s forecast failed for genre %s: %s", model_type, genre, exc)
            failed_genres.append(str(genre))
            continue

        forecast_parts.append(
            pd.DataFrame(
                {
                    "release_year": future_years,
                    "genre": genre,
                    "total_sales": result["predictions"],
                    "data_type": "Forecast",
                }
            )
        )

    if not historical_parts:
        return _error_figure("Not enough genre history for a forecast")

    genre_combined = pd.concat(historical_parts + forecast_parts, ignore_index=True)
    model_label = MODEL_LABELS.get(model_type, model_type)
    figure = px.line(
        genre_combined,
        x="release_year",
        y="total_sales",
        color="genre",
        line_dash="data_type",
        title=f"Genre Sales by Release Year Forecast ({model_label})",
        labels={"release_year": "Release Year", "total_sales": "Total Sales (millions)"},
    )
    figure.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        )
    )

    if failed_genres:
        figure.add_annotation(
            x=0.01,
            y=0.01,
            xref="paper",
            yref="paper",
            text="Forecast unavailable for: " + ", ".join(failed_genres),
            showarrow=False,
            font=dict(size=9),
        )

    return figure


def register_forecast_callbacks(app, df, df_cache, plotly_config=None):
    """Register forecast callbacks."""
    @app.callback(
        [
            Output("sales-forecast-chart", "figure"),
            Output("genre-forecast-chart", "figure"),
        ],
        [Input("forecast-button", "n_clicks")],
        [
            State("year-slider", "value"),
            State("platform-dropdown", "value"),
            State("console-gen-dropdown", "value"),
            State("genre-dropdown", "value"),
            State("publisher-dropdown", "value"),
            State("forecast-years", "value"),
            State("model-type", "value"),
        ],
        prevent_initial_call=True,
    )
    def generate_forecast(
        n_clicks,
        year_range,
        selected_platforms,
        selected_generations,
        selected_genres,
        selected_publishers,
        forecast_years,
        model_type,
    ):
        start_time = time.time()
        try:
            filtered_df = apply_filters(
                df,
                df_cache,
                year_range,
                selected_platforms,
                selected_generations,
                selected_genres,
                selected_publishers,
                [0, 10],
                None,
            )

            yearly_sales = (
                filtered_df.dropna(subset=["release_year"])
                .groupby("release_year", observed=False)["total_sales"]
                .sum()
                .reset_index()
                .sort_values("release_year")
            )

            if len(yearly_sales) < 5:
                empty = _error_figure("Not enough data for forecast (at least 5 years required)")
                return empty, empty

            forecast_years = max(1, int(forecast_years or 1))
            last_year = int(yearly_sales["release_year"].max())
            future_year_values = np.arange(last_year + 1, last_year + forecast_years + 1, dtype=float)
            years = yearly_sales["release_year"].to_numpy(dtype=float)
            sales = yearly_sales["total_sales"].to_numpy(dtype=float)

            validation_stats = evaluate_model_chronologically(model_type, years, sales)
            result = fit_and_predict_cached(
                model_type,
                tuple(map(float, years)),
                tuple(map(float, sales)),
                tuple(map(float, future_year_values)),
            )

            sales_figure = _build_sales_figure(
                yearly_sales,
                future_year_values,
                result,
                validation_stats,
                model_type,
            )
            genre_figure = _build_genre_figure(filtered_df, future_year_values, model_type)

            logger.info(
                "Forecast generation completed with %s in %.2fs",
                model_type,
                time.time() - start_time,
            )
            return sales_figure, genre_figure

        except (ForecastError, ValueError) as exc:
            logger.error("Requested %s forecast failed: %s", model_type, exc)
            error = _error_figure(f"{MODEL_LABELS.get(model_type, model_type)} forecast unavailable: {exc}")
            return error, error
        except Exception as exc:
            logger.exception("Unexpected forecast generation failure")
            error = _error_figure(f"Forecast generation failed: {exc}")
            return error, error
