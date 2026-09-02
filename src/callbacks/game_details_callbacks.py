"""Callbacks for the game details modal."""

import dash
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd

from src.utils.game_identity import match_game_from_chart_point


def _display_number(value, suffix=""):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}{suffix}"


def register_game_details_callbacks(app, df):
    """Register callbacks for selecting and displaying one exact game record."""

    @app.callback(
        Output("selected-game-data", "data"),
        [
            Input("top-games-bar", "clickData"),
            Input("critic-score-vs-sales", "clickData"),
            Input("sales-to-score-ratio", "clickData"),
        ],
        prevent_initial_call=True,
    )
    def capture_selected_game(top_games_click, critic_click, ratio_click):
        ctx = dash.callback_context
        if not ctx.triggered:
            return None

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        click_map = {
            "top-games-bar": top_games_click,
            "critic-score-vs-sales": critic_click,
            "sales-to-score-ratio": ratio_click,
        }
        click_data = click_map.get(trigger_id)
        if not click_data or not click_data.get("points"):
            return None

        return match_game_from_chart_point(df, trigger_id, click_data["points"][0])

    @app.callback(
        [
            Output("game-details-modal", "is_open"),
            Output("game-details-content", "children"),
        ],
        [
            Input("selected-game-data", "data"),
            Input("close-game-details", "n_clicks"),
        ],
        [State("game-details-modal", "is_open")],
        prevent_initial_call=True,
    )
    def toggle_modal(game_data, close_clicks, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open, []

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == "close-game-details":
            return False, []

        if trigger_id != "selected-game-data" or not game_data:
            return is_open, []

        release_date = game_data.get("release_date")
        if release_date is None or pd.isna(release_date):
            release_label = "N/A"
        else:
            release_label = str(release_date).split("T")[0].split(" ")[0]

        critic_score = game_data.get("critic_score")
        critic_label = (
            "N/A"
            if critic_score is None or pd.isna(critic_score)
            else f"{float(critic_score):.1f}/10"
        )
        sales_per_point = game_data.get("sales_per_point")
        efficiency_label = (
            "Not available"
            if sales_per_point is None or pd.isna(sales_per_point)
            else f"{float(sales_per_point):.2f} million sales per review point"
        )

        content = [
            html.H3(game_data.get("title", "N/A"), className="mb-3"),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P(
                                [
                                    html.Strong("Platform: "),
                                    html.Span(game_data.get("console", "N/A")),
                                ]
                            ),
                            html.P(
                                [
                                    html.Strong("Genre: "),
                                    html.Span(game_data.get("genre", "N/A")),
                                ]
                            ),
                            html.P(
                                [
                                    html.Strong("Publisher: "),
                                    html.Span(game_data.get("publisher", "N/A")),
                                ]
                            ),
                            html.P(
                                [
                                    html.Strong("Developer: "),
                                    html.Span(game_data.get("developer", "N/A")),
                                ]
                            ),
                            html.P(
                                [
                                    html.Strong("Release Date: "),
                                    html.Span(release_label),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.P(
                                [
                                    html.Strong("Total Sales: "),
                                    html.Span(
                                        _display_number(
                                            game_data.get("total_sales"),
                                            " million",
                                        )
                                    ),
                                ]
                            ),
                            html.P(
                                [
                                    html.Strong("Critic Score: "),
                                    html.Span(critic_label),
                                ]
                            ),
                            html.H5("Regional Sales Breakdown:", className="mt-3"),
                            html.P(
                                [
                                    html.Strong("North America: "),
                                    html.Span(
                                        _display_number(
                                            game_data.get("na_sales"),
                                            " million",
                                        )
                                    ),
                                ]
                            ),
                            html.P(
                                [
                                    html.Strong("Japan: "),
                                    html.Span(
                                        _display_number(
                                            game_data.get("jp_sales"),
                                            " million",
                                        )
                                    ),
                                ]
                            ),
                            html.P(
                                [
                                    html.Strong("Europe/Australia: "),
                                    html.Span(
                                        _display_number(
                                            game_data.get("pal_sales"),
                                            " million",
                                        )
                                    ),
                                ]
                            ),
                            html.P(
                                [
                                    html.Strong("Rest of World: "),
                                    html.Span(
                                        _display_number(
                                            game_data.get("other_sales"),
                                            " million",
                                        )
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
            html.Hr(),
            html.Div(
                [
                    html.H5("Sales Performance Analysis"),
                    html.P(f"Commercial Success Ratio: {efficiency_label}"),
                ],
                className="mt-3",
            ),
        ]
        return True, content
