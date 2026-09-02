"""Register all dashboard callback groups."""

import logging

from src.callbacks.comparison_callbacks import register_comparison_callbacks
from src.callbacks.export_callbacks import register_export_callbacks
from src.callbacks.forecast_callbacks import register_forecast_callbacks
from src.callbacks.game_details_callbacks import register_game_details_callbacks
from src.callbacks.graph_callbacks import register_graph_callbacks
from src.callbacks.seasonal_callbacks import register_seasonal_callbacks
from src.callbacks.theme_callbacks import register_theme_callbacks

logger = logging.getLogger(__name__)


def register_all_callbacks(app, df, df_cache, plotly_config=None):
    """Register every callback group used by the dashboard."""
    logger.info("Registering callbacks...")

    if plotly_config is None:
        plotly_config = {"use_custom_templates": True, "simple_charts": False}

    register_graph_callbacks(app, df, df_cache, plotly_config)
    register_export_callbacks(app, df, df_cache)
    register_game_details_callbacks(app, df)
    register_theme_callbacks(app)
    register_forecast_callbacks(app, df, df_cache, plotly_config)
    register_seasonal_callbacks(app, df, df_cache, plotly_config)
    register_comparison_callbacks(app, df, plotly_config)

    logger.info("All callbacks registered successfully.")
