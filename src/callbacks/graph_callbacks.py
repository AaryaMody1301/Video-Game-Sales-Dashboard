"""Callbacks and figure builders for the dashboard's core visualizations."""

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from src.data.data_loader import apply_filters

logger = logging.getLogger(__name__)

REGION_COLUMNS = ["na_sales", "jp_sales", "pal_sales", "other_sales"]
REGION_LABELS = {
    "na_sales": "North America",
    "jp_sales": "Japan",
    "pal_sales": "Europe/Australia",
    "other_sales": "Rest of World",
}
REGION_COLORS = {
    "na_sales": "blue",
    "jp_sales": "red",
    "pal_sales": "green",
    "other_sales": "orange",
}


def _display_limit(display_count):
    """Return a safe positive chart display limit."""
    try:
        return max(1, int(display_count))
    except (TypeError, ValueError):
        return 10


def _missing_platforms(filtered_df, selected_platforms):
    """Find selected platforms with no rows after all active filters are applied."""
    if not selected_platforms:
        return []

    present = set(filtered_df["console"].dropna().astype(str))
    missing = [platform for platform in selected_platforms if platform not in present]
    for platform in missing:
        logger.info("No rows match the current filters for platform: %s", platform)
    return missing


def _annotate_missing_platforms(fig, missing_platforms, y=0.9):
    """Annotate a figure when selected platforms have no matching rows."""
    if not missing_platforms:
        return fig

    fig.add_annotation(
        x=0.5,
        y=y,
        xref="paper",
        yref="paper",
        text=f"No rows match current filters for: {', '.join(missing_platforms)}",
        showarrow=False,
        font=dict(color="red", size=12),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="red",
        borderwidth=1,
        borderpad=4,
    )
    return fig


def _platform_figure(filtered_df, display_count, simple_charts, use_custom_templates):
    platform_sales = (
        filtered_df.groupby("console", observed=False)["total_sales"]
        .sum()
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(_display_limit(display_count))
    )
    title = f"Total Sales by Platform (Top {len(platform_sales)})"

    if simple_charts:
        fig = go.Figure(
            go.Bar(
                x=platform_sales["console"],
                y=platform_sales["total_sales"],
                text=platform_sales["total_sales"].round(2),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Platform",
            yaxis_title="Total Sales (millions)",
        )
        return fig

    fig = px.bar(
        platform_sales,
        x="console",
        y="total_sales",
        title=title,
        labels={"console": "Platform", "total_sales": "Total Sales (millions)"},
    )
    if use_custom_templates:
        fig.update_traces(
            marker_color=platform_sales["total_sales"],
            marker_colorscale="Viridis",
        )
    return fig


def _genre_figure(filtered_df, display_count, simple_charts, use_custom_templates):
    genre_sales = (
        filtered_df.groupby("genre", observed=False)["total_sales"]
        .sum()
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(_display_limit(display_count))
    )

    if simple_charts:
        fig = go.Figure(
            go.Pie(
                labels=genre_sales["genre"],
                values=genre_sales["total_sales"],
                hole=0.3,
            )
        )
        fig.update_layout(title="Sales Distribution by Genre")
        return fig

    fig = px.pie(
        genre_sales,
        values="total_sales",
        names="genre",
        title="Sales Distribution by Genre",
        hole=0.3,
    )
    if use_custom_templates:
        fig.update_traces(marker=dict(colors=px.colors.qualitative.Set3))
    return fig


def _release_year_figure(filtered_df, simple_charts):
    yearly_sales = (
        filtered_df.dropna(subset=["release_year"])
        .groupby("release_year", observed=False)["total_sales"]
        .sum()
        .reset_index()
        .sort_values("release_year")
    )
    title = "Lifetime Sales by Game Release Year"

    if simple_charts:
        fig = go.Figure(
            go.Scatter(
                x=yearly_sales["release_year"],
                y=yearly_sales["total_sales"],
                mode="lines+markers",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Release Year",
            yaxis_title="Total Sales (millions)",
        )
        return fig

    return px.line(
        yearly_sales,
        x="release_year",
        y="total_sales",
        title=title,
        labels={
            "release_year": "Release Year",
            "total_sales": "Total Sales (millions)",
        },
        markers=True,
    )


def _regional_release_year_figure(filtered_df, simple_charts, use_custom_templates):
    regional_yearly = (
        filtered_df.dropna(subset=["release_year"])
        .groupby("release_year", observed=False)[REGION_COLUMNS]
        .sum()
        .reset_index()
        .sort_values("release_year")
    )
    title = "Regional Lifetime Sales by Game Release Year"

    if simple_charts:
        fig = go.Figure()
        for column in REGION_COLUMNS:
            fig.add_trace(
                go.Scatter(
                    x=regional_yearly["release_year"],
                    y=regional_yearly[column],
                    mode="lines",
                    stackgroup="one",
                    name=REGION_LABELS[column],
                    line=dict(color=REGION_COLORS[column]),
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title="Release Year",
            yaxis_title="Sales (millions)",
        )
        return fig

    fig = px.area(
        regional_yearly,
        x="release_year",
        y=REGION_COLUMNS,
        title=title,
        labels={
            "release_year": "Release Year",
            "value": "Sales (millions)",
            "variable": "Region",
        },
    )
    if use_custom_templates:
        fig.update_traces(line=dict(width=0.5), selector=dict(type="scatter"))
        for index, column in enumerate(REGION_COLUMNS):
            fig.data[index].name = REGION_LABELS[column]
            fig.data[index].line.color = REGION_COLORS[column]
            fig.data[index].fillcolor = REGION_COLORS[column]
    return fig


def _top_games_figure(filtered_df, sort_method, display_count, simple_charts, use_custom_templates):
    sort_options = {
        "total_sales": ("total_sales", "Total Sales"),
        "critic_score": ("critic_score", "Critic Score"),
        "release_year": ("release_year", "Release Year"),
    }
    sort_column, sort_label = sort_options.get(
        sort_method,
        sort_options["total_sales"],
    )
    top_games = (
        filtered_df.dropna(subset=[sort_column])
        .sort_values(sort_column, ascending=False)
        .head(_display_limit(display_count))
    )
    title = f"Top {len(top_games)} Games by {sort_label}"

    if simple_charts:
        fig = go.Figure(
            go.Bar(
                x=top_games["total_sales"],
                y=top_games["title"],
                orientation="h",
                text=top_games["total_sales"].round(2),
                customdata=top_games[
                    ["release_year", "publisher", "critic_score", "console"]
                ],
                hovertemplate=(
                    "%{y}<br>Sales: %{x}<br>Year: %{customdata[0]}"
                    "<br>Publisher: %{customdata[1]}<br>Score: %{customdata[2]}"
                    "<br>Platform: %{customdata[3]}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Total Sales (millions)",
            yaxis_title="Game Title",
        )
        return fig

    fig = px.bar(
        top_games,
        x="total_sales",
        y="title",
        orientation="h",
        title=title,
        labels={"total_sales": "Total Sales (millions)", "title": "Game Title"},
        text="total_sales",
        hover_data=["release_year", "publisher", "critic_score", "console"],
    )
    if use_custom_templates and not top_games.empty:
        fig.update_traces(
            marker_color=top_games["genre"].astype("category").cat.codes,
            marker_colorscale="Viridis",
        )
    return fig


def _publisher_share_figure(filtered_df, display_count):
    publisher_sales = (
        filtered_df.groupby("publisher", observed=False)["total_sales"]
        .sum()
        .reset_index()
        .sort_values("total_sales", ascending=False)
    )
    limit = _display_limit(display_count)
    top_publishers = publisher_sales.head(limit).copy()
    other_sales = publisher_sales.iloc[limit:]["total_sales"].sum()

    if other_sales > 0:
        other_row = pd.DataFrame(
            {"publisher": ["Other"], "total_sales": [other_sales]}
        )
        top_publishers = pd.concat([top_publishers, other_row], ignore_index=True)

    return px.pie(
        top_publishers,
        values="total_sales",
        names="publisher",
        title=f"Publisher Share of Total Sales (Top {min(limit, len(publisher_sales))} + Other)",
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )


def _critic_sales_figure(filtered_df):
    scatter_df = filtered_df.dropna(subset=["critic_score", "total_sales"]).copy()
    scatter_df["size_value"] = scatter_df["total_sales"].clip(lower=0.01)

    fig = px.scatter(
        scatter_df,
        x="critic_score",
        y="total_sales",
        title="Critic Score vs. Total Sales",
        labels={
            "critic_score": "Critic Score",
            "total_sales": "Total Sales (millions)",
        },
        color="genre",
        size="size_value",
        hover_name="title",
        opacity=0.7,
        size_max=50,
        hover_data=["release_year", "publisher", "console"],
    )
    fig.update_layout(showlegend=False)
    return fig


def _regional_totals_figure(filtered_df):
    regional_totals = pd.DataFrame(
        {
            "region": [REGION_LABELS[column] for column in REGION_COLUMNS],
            "sales": [filtered_df[column].sum() for column in REGION_COLUMNS],
        }
    )
    return px.bar(
        regional_totals,
        x="region",
        y="sales",
        title="Sales by Region",
        labels={"region": "Region", "sales": "Total Sales (millions)"},
        color="region",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )


def _genre_release_year_figure(filtered_df):
    valid_rows = filtered_df.dropna(subset=["release_year"])
    top_genres = (
        valid_rows.groupby("genre", observed=False)["total_sales"]
        .sum()
        .nlargest(8)
        .index.tolist()
    )
    genre_yearly = (
        valid_rows[valid_rows["genre"].isin(top_genres)]
        .groupby(["release_year", "genre"], observed=False)["total_sales"]
        .sum()
        .reset_index()
    )
    fig = px.line(
        genre_yearly,
        x="release_year",
        y="total_sales",
        color="genre",
        title="Genre Lifetime Sales by Game Release Year",
        labels={
            "release_year": "Release Year",
            "total_sales": "Total Sales (millions)",
            "genre": "Genre",
        },
        line_shape="spline",
        render_mode="svg",
    )
    fig.update_layout(legend_title_text="Genre")
    return fig


def _console_generation_figure(filtered_df):
    generation_sales = (
        filtered_df.groupby("console_gen", observed=False)["total_sales"]
        .sum()
        .reset_index()
        .sort_values("total_sales", ascending=False)
    )
    fig = px.bar(
        generation_sales,
        x="console_gen",
        y="total_sales",
        title="Sales by Console Generation",
        labels={
            "console_gen": "Console Generation",
            "total_sales": "Total Sales (millions)",
        },
        color="console_gen",
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    generation_scores = (
        filtered_df.groupby("console_gen", observed=False)["critic_score"]
        .mean()
        .reset_index()
    )
    fig.add_trace(
        go.Scatter(
            x=generation_scores["console_gen"],
            y=generation_scores["critic_score"],
            name="Avg. Critic Score",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="red", width=3),
            marker=dict(size=10),
        )
    )
    fig.update_layout(
        yaxis2=dict(
            title="Average Critic Score",
            overlaying="y",
            side="right",
            range=[0, 10],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


def _publisher_performance_figure(filtered_df, display_count):
    top_publishers = (
        filtered_df["publisher"]
        .value_counts()
        .nlargest(_display_limit(display_count))
        .index.tolist()
    )
    publisher_data = filtered_df[filtered_df["publisher"].isin(top_publishers)]
    publisher_metrics = (
        publisher_data.groupby("publisher", observed=False)
        .agg(
            total_sales=("total_sales", "sum"),
            critic_score=("critic_score", "mean"),
            game_count=("title", "count"),
        )
        .reset_index()
    )
    return px.scatter(
        publisher_metrics,
        x="critic_score",
        y="total_sales",
        size="game_count",
        color="publisher",
        title="Publisher Performance Analysis",
        labels={
            "critic_score": "Average Critic Score",
            "total_sales": "Total Sales (millions)",
            "game_count": "Number of Games",
        },
        hover_data=["game_count"],
        size_max=60,
    )


def _sales_efficiency_figure(filtered_df, display_count):
    efficiency = filtered_df.dropna(subset=["sales_per_point"]).copy()
    top_efficiency = efficiency.nlargest(
        _display_limit(display_count),
        "sales_per_point",
    )
    fig = px.bar(
        top_efficiency,
        x="title",
        y="sales_per_point",
        title=f"Top {len(top_efficiency)} Games by Commercial Efficiency",
        labels={
            "title": "Game",
            "sales_per_point": "Sales per Review Point (millions)",
        },
        color="genre",
        hover_data=["total_sales", "critic_score", "release_year", "publisher"],
    )
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    return fig


def _build_figures(
    filtered_df,
    sort_method,
    display_count,
    simple_charts,
    use_custom_templates,
    missing_platforms,
):
    """Build the ordered figure list expected by the Dash callback outputs."""
    figures = [
        _platform_figure(
            filtered_df,
            display_count,
            simple_charts,
            use_custom_templates,
        ),
        _genre_figure(
            filtered_df,
            display_count,
            simple_charts,
            use_custom_templates,
        ),
        _release_year_figure(filtered_df, simple_charts),
        _regional_release_year_figure(
            filtered_df,
            simple_charts,
            use_custom_templates,
        ),
        _top_games_figure(
            filtered_df,
            sort_method,
            display_count,
            simple_charts,
            use_custom_templates,
        ),
        _publisher_share_figure(filtered_df, display_count),
        _critic_sales_figure(filtered_df),
        _regional_totals_figure(filtered_df),
        _genre_release_year_figure(filtered_df),
        _console_generation_figure(filtered_df),
        _publisher_performance_figure(filtered_df, display_count),
        _sales_efficiency_figure(filtered_df, display_count),
    ]
    return [
        _annotate_missing_platforms(figure, missing_platforms)
        for figure in figures
    ]


def register_graph_callbacks(app, df, df_cache, plotly_config=None):
    """Register callbacks for the dashboard's core visualizations."""
    config = plotly_config or {
        "use_custom_templates": True,
        "simple_charts": False,
    }
    use_custom_templates = config.get("use_custom_templates", True)
    simple_charts = config.get("simple_charts", False)

    @app.callback(
        [
            Output("sales-by-platform", "figure"),
            Output("sales-by-genre", "figure"),
            Output("sales-over-time", "figure"),
            Output("regional-sales-over-time", "figure"),
            Output("top-games-bar", "figure"),
            Output("publisher-market-share", "figure"),
            Output("critic-score-vs-sales", "figure"),
            Output("regional-sales-comparison", "figure"),
            Output("genre-trends-over-time", "figure"),
            Output("console-generation-comparison", "figure"),
            Output("publisher-performance", "figure"),
            Output("sales-to-score-ratio", "figure"),
        ],
        [
            Input("year-slider", "value"),
            Input("platform-dropdown", "value"),
            Input("console-gen-dropdown", "value"),
            Input("genre-dropdown", "value"),
            Input("publisher-dropdown", "value"),
            Input("critic-score-slider", "value"),
            Input("sort-method", "value"),
            Input("display-count", "value"),
            Input("search-bar", "value"),
        ],
    )
    def update_graphs(
        year_range,
        selected_platforms,
        selected_generations,
        selected_genres,
        selected_publishers,
        critic_range,
        sort_method,
        display_count,
        search_value,
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
            search_value,
        )
        missing = _missing_platforms(filtered_df, selected_platforms)
        return _build_figures(
            filtered_df,
            sort_method,
            display_count,
            simple_charts,
            use_custom_templates,
            missing,
        )
