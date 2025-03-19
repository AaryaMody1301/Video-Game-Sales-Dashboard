"""
Main entry point for the Video Game Sales Dashboard application
"""
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import logging
import os

# Import app components
from src.layouts.main_layout import create_layout
from src.data.data_loader import load_data
from src.utils.cache import DataFrameCache
from src.callbacks.register_callbacks import register_all_callbacks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Available themes
THEMES = {
    "Light": dbc.themes.BOOTSTRAP,
    "Dark": dbc.themes.DARKLY,
    "Slate": dbc.themes.SLATE,
    "Superhero": dbc.themes.SUPERHERO
}

# Initialize the Dash app with theme selector
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "Video Game Sales Dashboard"

# Load the data
df = load_data()

# Initialize the cache
df_cache = DataFrameCache(max_size=20)

# Set up the layout
app.layout = create_layout(df)

# Register callbacks
register_all_callbacks(app, df, df_cache)

def main():
    """Run the application server"""
    print("Starting dashboard server...")
    app.run(debug=True)

if __name__ == '__main__':
    main() 