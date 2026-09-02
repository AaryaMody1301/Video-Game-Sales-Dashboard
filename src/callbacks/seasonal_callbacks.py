"""Callbacks for release-season analysis."""

from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

from src.data.data_loader import apply_filters

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def register_seasonal_callbacks(app, df, df_cache, plotly_config=None):
    """Register charts that compare lifetime sales by game release season."""

    @app.callback(
        [
            Output("seasonal-sales-chart", "figure"),
            Output("monthly-sales-heatmap", "figure"),
            Output("quarterly-genre-distribution", "figure"),
            Output("seasonal-insights-text", "children"),
        ],
        [
            Input("year-slider", "value"),
            Input("platform-dropdown", "value"),
            Input("console-gen-dropdown", "value"),
            Input("genre-dropdown", "value"),
            Input("publisher-dropdown", "value"),
            Input("critic-score-slider", "value"),
        ],
    )
    def update_release_season_analysis(
        year_range,
        selected_platforms,
        selected_generations,
        selected_genres,
        selected_publishers,
        critic_range,
    ):
        filtered_df = apply_filters(
            df,
            df_cache,
            year_range,
            selected_platforms,
            selected_generations,
            selected_genres,
            selected_publishers,
            critic_range,
            None,
        )

        release_df = filtered_df.dropna(
            subset=["release_date", "release_month", "release_quarter", "total_sales"]
        ).copy()
        if release_df.empty:
            empty = px.bar(title="No release-season data available for the current filters")
            return empty, empty, empty, [html.P("No release-season data available.")]

        monthly = (
            release_df.groupby("release_month", observed=False)["total_sales"]
            .sum()
            .reset_index()
            .sort_values("release_month")
        )
        monthly["month_name"] = monthly["release_month"].map(MONTH_NAMES)

        monthly_chart = px.bar(
            monthly,
            x="month_name",
            y="total_sales",
            title="Lifetime Sales of Games by Release Month",
            labels={
                "month_name": "Release Month",
                "total_sales": "Reported Lifetime Sales (millions)",
            },
        )

        year_month = (
            release_df.groupby(["release_year", "release_month"], observed=False)["total_sales"]
            .sum()
            .reset_index()
        )
        pivot = year_month.pivot(
            index="release_year",
            columns="release_month",
            values="total_sales",
        ).fillna(0)
        heatmap = px.imshow(
            pivot,
            labels={
                "x": "Release Month",
                "y": "Release Year",
                "color": "Lifetime Sales (millions)",
            },
            x=[MONTH_NAMES[int(month)] for month in pivot.columns],
            y=pivot.index,
            title="Lifetime Sales by Release Year and Month",
            color_continuous_scale="Viridis",
        )

        top_genres = (
            release_df.groupby("genre", observed=False)["total_sales"]
            .sum()
            .nlargest(5)
            .index
        )
        quarter_genre = (
            release_df[release_df["genre"].isin(top_genres)]
            .groupby(["release_quarter", "genre"], observed=False)["total_sales"]
            .sum()
            .reset_index()
        )
        quarter_chart = px.bar(
            quarter_genre,
            x="release_quarter",
            y="total_sales",
            color="genre",
            title="Lifetime Sales by Release Quarter and Genre",
            labels={
                "release_quarter": "Release Quarter",
                "total_sales": "Reported Lifetime Sales (millions)",
                "genre": "Genre",
            },
            barmode="group",
        )

        peak_month = monthly.loc[monthly["total_sales"].idxmax()]
        total_sales = float(release_df["total_sales"].sum())
        q4_sales = float(
            release_df.loc[release_df["release_quarter"] == 4, "total_sales"].sum()
        )
        q4_share = q4_sales / total_sales * 100 if total_sales else 0.0

        insights = [
            html.P(
                f"Games released in {peak_month['month_name']} account for the highest combined "
                f"reported lifetime sales ({peak_month['total_sales']:.1f} million)."
            ),
            html.P(
                f"Games released in Q4 account for {q4_share:.1f}% of reported lifetime sales "
                "within the current filters."
            ),
            html.Small(
                "This dataset contains cumulative game sales and release dates, not transaction-level "
                "monthly sales. These charts describe release-season performance, not when purchases occurred.",
                className="text-muted",
            ),
        ]
        return monthly_chart, heatmap, quarter_chart, insights
