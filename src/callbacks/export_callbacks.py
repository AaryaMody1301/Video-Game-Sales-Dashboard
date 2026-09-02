"""Callbacks for data export functionality."""

from datetime import datetime

from dash import dcc
from dash.dependencies import Input, Output, State

from src.data.data_loader import apply_filters
from src.utils.exporting import serialize_dataframe


def register_export_callbacks(app, df, df_cache):
    """Register CSV/Excel export callbacks."""
    @app.callback(
        Output("download-dataframe-csv", "data"),
        [Input("export-button", "n_clicks")],
        [
            State("export-format-dropdown", "value"),
            State("year-slider", "value"),
            State("platform-dropdown", "value"),
            State("console-gen-dropdown", "value"),
            State("genre-dropdown", "value"),
            State("publisher-dropdown", "value"),
            State("critic-score-slider", "value"),
            State("search-bar", "value"),
        ],
        prevent_initial_call=True,
    )
    def export_data(
        n_clicks,
        export_format,
        year_range,
        selected_platforms,
        selected_generations,
        selected_genres,
        selected_publishers,
        critic_range,
        search_value,
    ):
        if not n_clicks:
            return None

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

        payload, extension = serialize_dataframe(filtered_df, export_format)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_game_sales_{timestamp}.{extension}"
        return dcc.send_bytes(payload, filename)
