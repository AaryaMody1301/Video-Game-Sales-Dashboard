"""
Main callback registration module
"""
import logging

# Import callback modules
from src.callbacks.graph_callbacks import register_graph_callbacks
from src.callbacks.export_callbacks import register_export_callbacks
from src.callbacks.game_details_callbacks import register_game_details_callbacks
from src.callbacks.theme_callbacks import register_theme_callbacks
from src.callbacks.forecast_callbacks import register_forecast_callbacks
from src.callbacks.seasonal_callbacks import register_seasonal_callbacks
from src.callbacks.comparison_callbacks import register_comparison_callbacks

logger = logging.getLogger(__name__)

def register_all_callbacks(app, df, df_cache):
    """
    Register all dashboard callbacks
    
    Args:
        app (dash.Dash): The Dash application
        df (pandas.DataFrame): The complete dataframe
        df_cache (DataFrameCache): Cache for filtered dataframes
    """
    logger.info("Registering callbacks...")
    
    # Register callback groups
    register_graph_callbacks(app, df, df_cache)
    register_export_callbacks(app, df, df_cache)
    register_game_details_callbacks(app, df)
    register_theme_callbacks(app)
    register_forecast_callbacks(app, df, df_cache)
    register_seasonal_callbacks(app, df, df_cache)
    register_comparison_callbacks(app, df)
    
    logger.info("All callbacks registered successfully.")