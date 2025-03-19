"""
Filter panel component for the dashboard
"""
from dash import dcc, html
import dash_bootstrap_components as dbc

def create_filter_panel(df):
    """
    Create the filter panel for the dashboard
    
    Args:
        df (pandas.DataFrame): The complete dataframe
        
    Returns:
        dash component: The filter panel
    """
    filter_panel = html.Div([
        html.H4("Filters", className="mt-3"),
        html.Label("Year Range:"),
        dcc.RangeSlider(
            id='year-slider',
            min=df['release_year'].min() if not df['release_year'].min() != None else 1980,
            max=df['release_year'].max() if not df['release_year'].max() != None else 2023,
            value=[1990, 2020],
            marks={str(year): str(year) for year in range(
                int(df['release_year'].min() if not df['release_year'].min() != None else 1980), 
                int(df['release_year'].max() if not df['release_year'].max() != None else 2023), 
                5)},
            step=1
        ),
        
        html.Label("Select Platform:", className="mt-3"),
        dcc.Dropdown(
            id='platform-dropdown',
            options=[{'label': platform, 'value': platform} 
                     for platform in sorted(df['console'].unique())],
            value=[],
            multi=True
        ),
        
        html.Label("Select Console Generation:", className="mt-3"),
        dcc.Dropdown(
            id='console-gen-dropdown',
            options=[{'label': gen, 'value': gen} 
                     for gen in sorted(df['console_gen'].unique())],
            value=[],
            multi=True
        ),
        
        html.Label("Select Genre:", className="mt-3"),
        dcc.Dropdown(
            id='genre-dropdown',
            options=[{'label': genre, 'value': genre} 
                     for genre in sorted(df['genre'].unique())],
            value=[],
            multi=True
        ),
        
        html.Label("Select Publisher:", className="mt-3"),
        dcc.Dropdown(
            id='publisher-dropdown',
            options=[{'label': publisher, 'value': publisher} 
                     for publisher in sorted(df['publisher'].unique())[:50]],  # Limiting to top 50 publishers
            value=[],
            multi=True
        ),
        
        html.Label("Critic Score Range:", className="mt-3"),
        dcc.RangeSlider(
            id='critic-score-slider',
            min=0,
            max=10,
            value=[0, 10],
            marks={i: str(i) for i in range(0, 11)},
            step=0.5
        ),
        
        html.Hr(),
        html.H5("Advanced Options"),
        
        html.Label("Sort Method:", className="mt-2"),
        dcc.RadioItems(
            id='sort-method',
            options=[
                {'label': 'Total Sales', 'value': 'total_sales'},
                {'label': 'Critic Score', 'value': 'critic_score'},
                {'label': 'Release Year', 'value': 'release_year'}
            ],
            value='total_sales',
            labelStyle={'display': 'block', 'margin-bottom': '5px'}
        ),
        
        html.Label("Display Count:", className="mt-2"),
        dcc.Slider(
            id='display-count',
            min=5,
            max=25,
            value=10,
            marks={i: str(i) for i in [5, 10, 15, 20, 25]},
            step=5
        ),
        
        html.Label("Search Game Title:", className="mt-3"),
        dcc.Input(id='search-bar', type='text', placeholder='Enter game title...', debounce=True),
        
        html.Label("Export Format:", className="mt-3"),
        dcc.Dropdown(
            id='export-format-dropdown',
            options=[
                {'label': 'CSV', 'value': 'csv'},
                {'label': 'Excel', 'value': 'excel'},
                {'label': 'PDF', 'value': 'pdf'}
            ],
            value='csv',
            clearable=False
        ),
        
        html.Button("Export Data", id="export-button", className="mt-3 btn btn-primary"),
        dcc.Download(id="download-dataframe-csv")
    ])
    
    return filter_panel 