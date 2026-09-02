"""Callbacks for game comparison functionality."""

from dash import dash_table, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.game_identity import add_game_ids, game_label, select_games_by_id


MAX_COMPARISON_GAMES = 5


def _empty_comparison(message):
    figure = px.bar(title=message)
    return [
        figure,
        px.bar(title=message),
        px.bar(title=message),
        html.Div(),
        html.P(message, className="text-danger"),
    ]


def register_comparison_callbacks(app, df, plotly_config=None):
    """Register game comparison callbacks using unique game identities."""
    games = add_game_ids(df)

    @app.callback(
        Output("game-comparison-dropdown", "options"),
        [Input("search-bar", "value")],
    )
    def update_game_dropdown(search_term):
        candidates = games
        if search_term and search_term.strip():
            candidates = candidates[
                candidates["title"].str.contains(
                    search_term.strip(),
                    case=False,
                    na=False,
                    regex=False,
                )
            ].nlargest(20, "total_sales")
        else:
            candidates = candidates.nlargest(50, "total_sales")

        return [
            {"label": game_label(row), "value": row.game_id}
            for row in candidates.itertuples(index=False)
        ]

    @app.callback(
        [
            Output("comparison-sales-chart", "figure"),
            Output("comparison-regional-chart", "figure"),
            Output("comparison-metrics-chart", "figure"),
            Output("comparison-table", "children"),
            Output("comparison-message", "children"),
        ],
        [Input("compare-button", "n_clicks")],
        [State("game-comparison-dropdown", "value")],
        prevent_initial_call=True,
    )
    def compare_games(n_clicks, selected_game_ids):
        if not selected_game_ids or len(selected_game_ids) < 2:
            return _empty_comparison("Select at least two games to compare")

        if len(selected_game_ids) > MAX_COMPARISON_GAMES:
            return _empty_comparison(
                f"Select no more than {MAX_COMPARISON_GAMES} games for a readable comparison"
            )

        comparison_df = select_games_by_id(df, selected_game_ids)
        if len(comparison_df) != len(set(selected_game_ids)):
            return _empty_comparison("One or more selected games were not found in the dataset")

        comparison_df["display_name"] = [
            game_label(row) for row in comparison_df.itertuples(index=False)
        ]

        fig_sales = px.bar(
            comparison_df,
            x="display_name",
            y="total_sales",
            title="Total Sales Comparison",
            labels={"display_name": "Game", "total_sales": "Total Sales (millions)"},
            color="genre",
            hover_data=["publisher", "console", "release_year"],
            text="total_sales",
        )
        fig_sales.update_traces(texttemplate="%{text:.1f}M", textposition="outside")

        regional_data = pd.melt(
            comparison_df,
            id_vars=["display_name"],
            value_vars=["na_sales", "jp_sales", "pal_sales", "other_sales"],
            var_name="region",
            value_name="sales",
        )
        regional_data["region"] = regional_data["region"].map(
            {
                "na_sales": "North America",
                "jp_sales": "Japan",
                "pal_sales": "Europe/Australia",
                "other_sales": "Rest of World",
            }
        )
        fig_regional = px.bar(
            regional_data,
            x="display_name",
            y="sales",
            color="region",
            title="Regional Sales Breakdown",
            labels={"display_name": "Game", "sales": "Sales (millions)", "region": "Region"},
            barmode="group",
        )

        max_sales = comparison_df["total_sales"].max()
        metrics = comparison_df.copy()
        metrics["normalized_sales"] = (
            metrics["total_sales"] / max_sales * 100 if max_sales > 0 else 0.0
        )
        metrics["normalized_critic"] = metrics["critic_score"] / 10 * 100

        for region in ("na", "jp", "pal"):
            metrics[f"{region}_percent_display"] = np.where(
                metrics["total_sales"] > 0,
                metrics[f"{region}_sales"] / metrics["total_sales"] * 100,
                0.0,
            )

        fig_metrics = go.Figure()
        for row in metrics.itertuples(index=False):
            critic_value = None if pd.isna(row.normalized_critic) else float(row.normalized_critic)
            fig_metrics.add_trace(
                go.Scatterpolar(
                    r=[
                        float(row.normalized_sales),
                        critic_value,
                        float(row.na_percent_display),
                        float(row.jp_percent_display),
                        float(row.pal_percent_display),
                    ],
                    theta=["Total Sales", "Critic Score", "NA Market", "JP Market", "EU Market"],
                    fill="toself",
                    name=row.display_name,
                )
            )

        fig_metrics.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Game Metrics Comparison",
        )

        table_df = comparison_df[
            [
                "display_name",
                "publisher",
                "genre",
                "release_year",
                "total_sales",
                "critic_score",
            ]
        ].copy()
        table_df["total_sales"] = table_df["total_sales"].map(lambda value: f"{value:.2f} M")
        table_df["critic_score"] = table_df["critic_score"].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.1f}/10"
        )

        comparison_table = dash_table.DataTable(
            id="game-metrics-table",
            columns=[
                {"name": column.replace("_", " ").title(), "id": column}
                for column in table_df.columns
            ],
            data=table_df.to_dict("records"),
            style_cell={
                "textAlign": "left",
                "padding": "5px",
                "whiteSpace": "normal",
                "height": "auto",
            },
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "rgb(248, 248, 248)",
                }
            ],
        )

        message = html.P(
            f"Comparing {len(selected_game_ids)} games.",
            className="text-success",
        )
        return fig_sales, fig_regional, fig_metrics, comparison_table, message
