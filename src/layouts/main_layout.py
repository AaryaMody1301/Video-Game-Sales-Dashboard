"""
Main layout for the Video Game Sales Dashboard
"""
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from src.components.filters import create_filter_panel
from src.components.tabs import create_tab_content
from src.components.modals import create_game_details_modal

def create_layout(df):
    """
    Create the main layout for the dashboard
    
    Args:
        df (pandas.DataFrame): The complete dataframe
        
    Returns:
        dash component: The main layout
    """
    layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Video Game Sales Dashboard", className="text-center mb-4"),
                html.P("Explore global video game sales data across platforms, genres, and time periods", 
                      className="text-center")
            ], width=10),
            dbc.Col([
                html.Label("Theme:", className="mr-2"),
                dcc.Dropdown(
                    id="theme-selector",
                    options=[
                        {"label": "Light", "value": "Light"},
                        {"label": "Dark", "value": "Dark"},
                        {"label": "Slate", "value": "Slate"},
                        {"label": "Superhero", "value": "Superhero"}
                    ],
                    value="Light",
                    clearable=False,
                    style={"width": "150px"}
                )
            ], width=2, className="d-flex align-items-center")
        ]),
        
        # Main content
        dbc.Row([
            # Filters panel
            dbc.Col([
                create_filter_panel(df)
            ], width=3),
            
            # Visualization tabs
            dbc.Col([
                create_tab_content()
            ], width=9),
        ]),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Dataset Information", className="mt-4"),
                    html.P(f"Total records: {len(df)}"),
                    html.P(f"Date range: {df['release_year'].min()} - {df['release_year'].max()}"),
                    html.P(f"Platforms: {len(df['console'].unique())}"),
                    html.P(f"Genres: {len(df['genre'].unique())}"),
                    html.P(f"Publishers: {len(df['publisher'].unique())}"),
                    html.P("Dashboard created with Dash and Plotly", className="text-muted mt-3"),
                ])
            ], width=12)
        ]),
        
        # Add the modal to the layout
        create_game_details_modal(),
        
        # Store for selected game data
        dcc.Store(id='selected-game-data'),
        
        # Hidden div for storing the current theme
        html.Div(id="theme-div", style={"display": "none"}),
        
        # Store component to track the current theme
        dcc.Store(id="theme-store", data={"current_theme": "Light"})
    ], fluid=True)
    
    return layout 