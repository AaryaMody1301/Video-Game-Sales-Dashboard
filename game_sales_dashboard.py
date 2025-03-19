import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import os
import io
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
import xlsxwriter
import traceback
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging

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

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add this after the import statements, before the load_data function
from functools import lru_cache
import time

# Fix the DataFrameCache class implementation
class DataFrameCache:
    def __init__(self, max_size=10):
        self.cache = {}
        self.max_size = max_size
        self.timestamps = {}
    
    def get_key(self, filters):
        # Create a hashable key from the filters
        return str(hash(str(filters)))
    
    def get(self, filters):
        key = self.get_key(filters)
        if key in self.cache:
            # Update timestamp for LRU tracking
            self.timestamps[key] = time.time()
            logger.debug(f"Cache hit for filter set {key}")
            return self.cache[key]
        return None
    
    def set(self, filters, df):
        key = self.get_key(filters)
        
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            logger.debug(f"Cache full, removing {oldest_key}")
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        # Add new item to cache
        self.cache[key] = df.copy()  # Store a copy to avoid reference issues
        self.timestamps[key] = time.time()
        logger.debug(f"Added filtered dataframe to cache with key {key}")

# Initialize the cache
df_cache = DataFrameCache(max_size=20)

# Helper function to apply filters to dataframe (used by multiple callbacks)
def apply_filters(year_range, selected_platforms, selected_generations, selected_genres, 
                 selected_publishers, critic_range, search_value):
    # Create a unique key for this filter combination
    filters = (
        tuple(year_range), 
        tuple(selected_platforms) if selected_platforms else None,
        tuple(selected_generations) if selected_generations else None,
        tuple(selected_genres) if selected_genres else None,
        tuple(selected_publishers) if selected_publishers else None,
        tuple(critic_range),
        search_value
    )
    
    # Check if we have a cached result
    cached_df = df_cache.get(filters)
    if cached_df is not None:
        return cached_df
    
    # If not cached, filter the data
    filtered_df = df.copy()
    
    # Year range filter
    if not pd.isna(year_range[0]) and not pd.isna(year_range[1]):
        filtered_df = filtered_df[
            (filtered_df['release_year'] >= year_range[0]) & 
            (filtered_df['release_year'] <= year_range[1])
        ]
    
    # Platform filter
    if selected_platforms:
        filtered_df = filtered_df[filtered_df['console'].isin(selected_platforms)]
    
    # Console generation filter
    if selected_generations:
        filtered_df = filtered_df[filtered_df['console_gen'].isin(selected_generations)]
    
    # Genre filter
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    # Publisher filter
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    
    # Critic score filter
    filtered_df = filtered_df[
        (filtered_df['critic_score'] >= critic_range[0]) & 
        (filtered_df['critic_score'] <= critic_range[1]) |
        (filtered_df['critic_score'].isna())  # Include games with no critic score
    ]
    
    # Search filter
    if search_value:
        filtered_df = filtered_df[filtered_df['title'].str.contains(search_value, case=False, na=False)]
    
    # Cache the result before returning
    df_cache.set(filters, filtered_df)
    
    return filtered_df

# Load and preprocess the data
def load_data():
    try:
        logger.info("Loading video game sales data...")
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path to the CSV file
        csv_path = os.path.join(script_dir, 'vgchartz-2024.csv')
        logger.info(f"Looking for data file at: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # Clean the data
        # Convert sales columns to numeric, replacing empty strings with NaN
        sales_columns = ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales']
        for col in sales_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert release_date to datetime, coercing errors to NaT
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        
        # Extract release year
        df['release_year'] = df['release_date'].dt.year
        
        # Fill NaN with 'Unknown' for categorical columns
        categorical_cols = ['console', 'genre', 'publisher', 'developer']
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        
        # Convert critic_score to numeric
        df['critic_score'] = pd.to_numeric(df['critic_score'], errors='coerce')
        
        # Additional data preprocessing
        # Create console generation categories (updated with newer consoles)
        console_generations = {
            # Fifth Generation (1993-2002)
            'PS1': 'Fifth Gen', 'PS': 'Fifth Gen', 'N64': 'Fifth Gen', 'SAT': 'Fifth Gen', 'DC': 'Fifth Gen',
            # Sixth Generation (1998-2009)
            'PS2': 'Sixth Gen', 'XB': 'Sixth Gen', 'GC': 'Sixth Gen', 
            # Seventh Generation (2005-2013)
            'PS3': 'Seventh Gen', 'X360': 'Seventh Gen', 'Wii': 'Seventh Gen',
            # Eighth Generation (2012-2020)
            'PS4': 'Eighth Gen', 'XOne': 'Eighth Gen', 'WiiU': 'Eighth Gen', 'NS': 'Eighth Gen',
            # Ninth Generation (2020-)
            'PS5': 'Ninth Gen', 'XSX': 'Ninth Gen', 'XS': 'Ninth Gen',
            # Handhelds
            'GBA': 'Handheld', 'DS': 'Handheld', 'PSP': 'Handheld', '3DS': 'Handheld', 'PSV': 'Handheld',
            'Switch': 'Hybrid',
            'PC': 'PC'
        }
        
        # Apply console generation mapping
        df['console_gen'] = df['console'].map(lambda x: console_generations.get(x, 'Other'))
        
        # Create decade column for timeline analysis
        df['decade'] = df['release_year'].apply(lambda x: f"{int(x/10)*10}s" if not pd.isna(x) else "Unknown")
        
        # Calculate commercial success ratio (total sales per critic score point)
        # Avoid division by zero by adding a small epsilon
        df['sales_per_point'] = df['total_sales'] / (df['critic_score'].replace(0, np.nan) + 1e-10)
        
        # Create publisher tier categories based on number of titles
        publisher_counts = df['publisher'].value_counts()
        top_publishers = publisher_counts[publisher_counts > 20].index.tolist()
        df['publisher_tier'] = df['publisher'].apply(lambda x: x if x in top_publishers else 'Other Publishers')
        
        # Calculate regional sales percentage (safely)
        df['total_sales_safe'] = df['total_sales'].replace(0, np.nan)
        df['na_percent'] = (df['na_sales'] / df['total_sales_safe'] * 100).round(1).fillna(0)
        df['jp_percent'] = (df['jp_sales'] / df['total_sales_safe'] * 100).round(1).fillna(0)
        df['pal_percent'] = (df['pal_sales'] / df['total_sales_safe'] * 100).round(1).fillna(0)
        df['other_percent'] = (df['other_sales'] / df['total_sales_safe'] * 100).round(1).fillna(0)
        df = df.drop('total_sales_safe', axis=1)
        
        # Additional metadata for analysis
        df['release_month'] = df['release_date'].dt.month
        df['release_quarter'] = df['release_date'].dt.quarter
        df['has_critic_score'] = ~df['critic_score'].isna()
        
        logger.info(f"Data loaded successfully. {len(df)} records found.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.debug(traceback.format_exc())
        # Return a minimal DataFrame with sample data to allow the app to start
        columns = ['title', 'console', 'publisher', 'developer', 'genre', 'release_date', 
                   'total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'critic_score']
        return pd.DataFrame(columns=columns)

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

# Define game details modal
game_details_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Game Details"), close_button=True),
        dbc.ModalBody([
            html.Div(id="game-details-content")
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-game-details", className="ml-auto")
        ),
    ],
    id="game-details-modal",
    size="lg",
)

# Define the layout
app.layout = dbc.Container([
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
                options=[{"label": theme, "value": theme} for theme in THEMES],
                value="Light",
                clearable=False,
                style={"width": "150px"}
            )
        ], width=2, className="d-flex align-items-center")
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Filters", className="mt-3"),
            html.Label("Year Range:"),
            dcc.RangeSlider(
                id='year-slider',
                min=df['release_year'].min() if not pd.isna(df['release_year'].min()) else 1980,
                max=df['release_year'].max() if not pd.isna(df['release_year'].max()) else 2023,
                value=[1990, 2020],
                marks={str(year): str(year) for year in range(
                    int(df['release_year'].min() if not pd.isna(df['release_year'].min()) else 1980), 
                    int(df['release_year'].max() if not pd.isna(df['release_year'].max()) else 2023), 
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
        ], width=3),
        
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='sales-by-platform')
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='sales-by-genre')
                        ], width=12),
                    ]),
                ], label="Market Overview"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='sales-over-time')
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='regional-sales-over-time')
                        ], width=12),
                    ]),
                ], label="Time Trends"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='top-games-bar')
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='publisher-market-share')
                        ], width=12),
                    ]),
                ], label="Top Performers"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='critic-score-vs-sales')
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='regional-sales-comparison')
                        ], width=12),
                    ]),
                ], label="Sales Analysis"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='genre-trends-over-time')
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='console-generation-comparison')
                        ], width=12),
                    ]),
                ], label="Trends Analysis"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='publisher-performance')
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='sales-to-score-ratio')
                        ], width=12),
                    ]),
                ], label="Publisher Insights"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H4("Sales Forecast", className="mt-3"),
                                html.P("Predict future trends based on historical data using machine learning models"),
                                html.Label("Forecast Years:"),
                                dcc.Slider(
                                    id='forecast-years',
                                    min=1,
                                    max=10,
                                    value=5,
                                    marks={i: str(i) for i in range(1, 11)},
                                    step=1
                                ),
                                html.Label("Prediction Model:", className="mt-3"),
                                dcc.RadioItems(
                                    id='model-type',
                                    options=[
                                        {'label': 'Linear Regression', 'value': 'linear'},
                                        {'label': 'Polynomial Regression', 'value': 'poly'}
                                    ],
                                    value='linear',
                                    labelStyle={'display': 'block', 'margin-bottom': '5px'}
                                ),
                                html.Button("Generate Forecast", id="forecast-button", 
                                           className="mt-3 btn btn-success"),
                            ], className="p-3 border rounded")
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='sales-forecast-chart')
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='genre-forecast-chart')
                        ], width=12)
                    ])
                ], label="Predictive Analytics"),

                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='seasonal-sales-chart')
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='monthly-sales-heatmap')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='quarterly-genre-distribution')
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Seasonal Insights", className="mt-3"),
                            html.Div(id="seasonal-insights-text", className="p-3 border rounded")
                        ], width=12),
                    ]),
                ], label="Seasonal Analysis"),

                # Add a new tab for game comparison after the "Seasonal Analysis" tab

                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Game Comparison", className="mt-3"),
                            html.P("Select games to compare their performance and metrics side by side"),
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Search and select games to compare:"),
                                            dcc.Dropdown(
                                                id='game-comparison-dropdown',
                                                options=[],  # Will be populated dynamically
                                                value=[],
                                                multi=True,
                                                placeholder="Type to search for games...",
                                            ),
                                        ], width=9),
                                        dbc.Col([
                                            html.Br(),
                                            dbc.Button(
                                                "Compare Games",
                                                id="compare-button",
                                                color="primary",
                                                className="mt-2"
                                            ),
                                        ], width=3, className="d-flex align-items-center"),
                                    ]),
                                    html.Div(id="comparison-message", className="mt-3 text-muted"),
                                ])
                            ])
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='comparison-sales-chart')
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='comparison-regional-chart')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='comparison-metrics-chart')
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='comparison-table', className="mt-3")
                        ], width=12)
                    ])
                ], label="Game Comparison"),
                
            ]),
        ], width=9),
    ]),
    
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
    game_details_modal,
    
    # Store for selected game data
    dcc.Store(id='selected-game-data'),
    
    # Hidden div for storing the current theme
    html.Div(id="theme-div", style={"display": "none"}),
    
    # Store component to track the current theme
    dcc.Store(id="theme-store", data={"current_theme": "Light"})
], fluid=True)

# Define callbacks for interactivity
@app.callback(
    [Output('sales-by-platform', 'figure'),
     Output('sales-by-genre', 'figure'),
     Output('sales-over-time', 'figure'),
     Output('regional-sales-over-time', 'figure'),
     Output('top-games-bar', 'figure'),
     Output('publisher-market-share', 'figure'),
     Output('critic-score-vs-sales', 'figure'),
     Output('regional-sales-comparison', 'figure'),
     Output('genre-trends-over-time', 'figure'),
     Output('console-generation-comparison', 'figure'),
     Output('publisher-performance', 'figure'),
     Output('sales-to-score-ratio', 'figure')],
    [Input('year-slider', 'value'),
     Input('platform-dropdown', 'value'),
     Input('console-gen-dropdown', 'value'),
     Input('genre-dropdown', 'value'),
     Input('publisher-dropdown', 'value'),
     Input('critic-score-slider', 'value'),
     Input('sort-method', 'value'),
     Input('display-count', 'value'),
     Input('search-bar', 'value')]
)
def update_graphs(year_range, selected_platforms, selected_generations, selected_genres, 
                  selected_publishers, critic_range, sort_method, display_count, search_value):
    # Filter data based on user selections
    filtered_df = apply_filters(year_range, selected_platforms, selected_generations, selected_genres, 
                                selected_publishers, critic_range, search_value)
    
    # Sales by platform chart
    platform_sales = filtered_df.groupby('console')['total_sales'].sum().reset_index()
    platform_sales = platform_sales.sort_values('total_sales', ascending=False).head(display_count)
    
    fig_platform = px.bar(
        platform_sales, 
        x='console', 
        y='total_sales',
        title=f'Total Sales by Platform (Top {len(platform_sales)})',
        labels={'console': 'Platform', 'total_sales': 'Total Sales (millions)'},
        color='total_sales',
        color_continuous_scale='Viridis'
    )
    
    # Sales by genre chart
    genre_sales = filtered_df.groupby('genre')['total_sales'].sum().reset_index()
    genre_sales = genre_sales.sort_values('total_sales', ascending=False)
    
    fig_genre = px.pie(
        genre_sales, 
        values='total_sales', 
        names='genre',
        title='Sales Distribution by Genre',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Sales over time chart
    yearly_sales = filtered_df.groupby('release_year')['total_sales'].sum().reset_index()
    yearly_sales = yearly_sales.sort_values('release_year')
    
    fig_time = px.line(
        yearly_sales, 
        x='release_year', 
        y='total_sales',
        title='Video Game Sales Trend Over Time',
        labels={'release_year': 'Year', 'total_sales': 'Total Sales (millions)'},
        markers=True
    )
    
    # Regional sales over time
    regional_yearly = filtered_df.groupby('release_year')[
        ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']
    ].sum().reset_index()
    
    fig_regional_time = px.area(
        regional_yearly, 
        x='release_year', 
        y=['na_sales', 'jp_sales', 'pal_sales', 'other_sales'],
        title='Regional Sales Over Time',
        labels={
            'release_year': 'Year', 
            'value': 'Sales (millions)',
            'variable': 'Region'
        },
        color_discrete_map={
            'na_sales': 'blue',
            'jp_sales': 'red',
            'pal_sales': 'green',
            'other_sales': 'orange'
        }
    )
    
    # Top games bar chart
    if sort_method == 'total_sales':
        sort_col = 'total_sales'
        sort_label = 'Total Sales'
    elif sort_method == 'critic_score':
        sort_col = 'critic_score'
        sort_label = 'Critic Score'
    else:
        sort_col = 'release_year'
        sort_label = 'Release Year'
    
    top_games = filtered_df.sort_values(sort_col, ascending=False)
    top_games = top_games.dropna(subset=[sort_col]).head(display_count)
    
    fig_top_games = px.bar(
        top_games,
        x='total_sales',
        y='title',
        orientation='h',
        title=f'Top {len(top_games)} Games by {sort_label}',
        labels={'total_sales': 'Total Sales (millions)', 'title': 'Game Title'},
        color='genre',
        text='total_sales',
        hover_data=['release_year', 'publisher', 'critic_score']
    )
    
    fig_top_games.update_traces(texttemplate='%{text:.1f}M', textposition='outside')
    fig_top_games.update_layout(yaxis={'categoryorder':'total ascending'})
    
    # Publisher market share
    publisher_sales = filtered_df.groupby('publisher')['total_sales'].sum().reset_index()
    publisher_sales = publisher_sales.sort_values('total_sales', ascending=False).head(display_count)
    
    fig_publisher = px.pie(
        publisher_sales, 
        values='total_sales', 
        names='publisher',
        title=f'Top {len(publisher_sales)} Publishers by Market Share',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Critic score vs sales
    scatter_df = filtered_df.dropna(subset=['critic_score']).copy()
    # Handle NaN values in total_sales by replacing them with a default size value
    scatter_df['size_value'] = scatter_df['total_sales'].fillna(1)  # Replace NaN with 1 for sizing
    
    fig_critic = px.scatter(
        scatter_df,
        x='critic_score',
        y='total_sales',
        title='Critic Score vs. Total Sales',
        labels={'critic_score': 'Critic Score', 'total_sales': 'Total Sales (millions)'},
        color='genre',
        size='size_value',  # Use the cleaned size column
        hover_name='title',
        opacity=0.7,
        size_max=50,
        hover_data=['release_year', 'publisher', 'console']
    )
    
    # Add trendline
    fig_critic.update_layout(showlegend=False)
    
    # Regional sales comparison
    regional_data = pd.melt(
        filtered_df,
        id_vars=['title'],
        value_vars=['na_sales', 'jp_sales', 'pal_sales', 'other_sales'],
        var_name='region',
        value_name='sales'
    )
    
    region_labels = {
        'na_sales': 'North America', 
        'jp_sales': 'Japan', 
        'pal_sales': 'Europe/Australia', 
        'other_sales': 'Rest of World'
    }
    
    regional_data['region'] = regional_data['region'].map(region_labels)
    
    regional_totals = regional_data.groupby('region')['sales'].sum().reset_index()
    
    fig_regional = px.bar(
        regional_totals,
        x='region',
        y='sales',
        title='Sales by Region',
        labels={'region': 'Region', 'sales': 'Total Sales (millions)'},
        color='region',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # NEW VISUALIZATIONS
    
    # 1. Genre trends over time
    genre_yearly = filtered_df.groupby(['release_year', 'genre'])['total_sales'].sum().reset_index()
    genre_yearly = genre_yearly[~genre_yearly['release_year'].isna()]
    
    # Filter to top genres for clarity
    top_genres = filtered_df.groupby('genre')['total_sales'].sum().nlargest(8).index.tolist()
    genre_yearly_filtered = genre_yearly[genre_yearly['genre'].isin(top_genres)]
    
    fig_genre_trends = px.line(
        genre_yearly_filtered,
        x='release_year',
        y='total_sales',
        color='genre',
        title='Genre Popularity Trends Over Time',
        labels={'release_year': 'Year', 'total_sales': 'Total Sales (millions)', 'genre': 'Genre'},
        line_shape='spline',
        render_mode='svg'
    )
    
    fig_genre_trends.update_layout(legend_title_text='Genre')
    
    # 2. Console generation comparison
    gen_sales = filtered_df.groupby('console_gen')['total_sales'].sum().reset_index()
    gen_sales = gen_sales.sort_values('total_sales', ascending=False)
    
    fig_console_gen = px.bar(
        gen_sales,
        x='console_gen',
        y='total_sales',
        title='Sales by Console Generation',
        labels={'console_gen': 'Console Generation', 'total_sales': 'Total Sales (millions)'},
        color='console_gen',
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    
    # Add average critic score on secondary y-axis
    gen_scores = filtered_df.groupby('console_gen')['critic_score'].mean().reset_index()
    gen_scores = gen_scores.sort_values('console_gen')
    
    fig_console_gen.add_trace(
        go.Scatter(
            x=gen_scores['console_gen'],
            y=gen_scores['critic_score'],
            name='Avg. Critic Score',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        )
    )
    
    fig_console_gen.update_layout(
        yaxis2=dict(
            title='Average Critic Score',
            overlaying='y',
            side='right',
            range=[0, 10]
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # 3. Publisher performance analysis
    # Get top publishers by game count
    top_n_publishers = filtered_df['publisher'].value_counts().nlargest(display_count).index.tolist()
    publisher_data = filtered_df[filtered_df['publisher'].isin(top_n_publishers)]
    
    # Calculate metrics for each publisher
    publisher_metrics = publisher_data.groupby('publisher').agg({
        'total_sales': 'sum',
        'critic_score': 'mean',
        'title': 'count'
    }).reset_index()
    
    publisher_metrics.rename(columns={'title': 'game_count'}, inplace=True)
    
    fig_publisher_perf = px.scatter(
        publisher_metrics,
        x='critic_score',
        y='total_sales',
        size='game_count',
        color='publisher',
        title='Publisher Performance Analysis',
        labels={
            'critic_score': 'Average Critic Score', 
            'total_sales': 'Total Sales (millions)',
            'game_count': 'Number of Games'
        },
        hover_data=['game_count'],
        size_max=60
    )
    
    fig_publisher_perf.update_layout(showlegend=True)
    
    # 4. Sales to score ratio (commercial success per review point)
    # Calculate commercial efficiency
    success_ratio = filtered_df.dropna(subset=['critic_score', 'total_sales']).copy()
    success_ratio['sales_per_point'] = success_ratio['total_sales'] / success_ratio['critic_score']
    
    # Get top games by sales_per_point ratio
    top_efficiency = success_ratio.nlargest(display_count, 'sales_per_point')
    
    fig_ratio = px.bar(
        top_efficiency,
        x='title',
        y='sales_per_point',
        title=f'Top {len(top_efficiency)} Games by Commercial Efficiency (Sales per Review Point)',
        labels={
            'title': 'Game', 
            'sales_per_point': 'Sales per Review Point (millions)',
        },
        color='genre',
        hover_data=['total_sales', 'critic_score', 'release_year', 'publisher']
    )
    
    fig_ratio.update_layout(xaxis={'categoryorder': 'total descending'})
    
    return [fig_platform, fig_genre, fig_time, fig_regional_time, 
            fig_top_games, fig_publisher, fig_critic, fig_regional,
            fig_genre_trends, fig_console_gen, fig_publisher_perf, fig_ratio]

@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("export-button", "n_clicks"),
     Input("export-format-dropdown", "value")],
    [State('year-slider', 'value'),
     State('platform-dropdown', 'value'),
     State('console-gen-dropdown', 'value'),
     State('genre-dropdown', 'value'),
     State('publisher-dropdown', 'value'),
     State('critic-score-slider', 'value'),
     State('search-bar', 'value')],
    prevent_initial_call=True,
)
def export_data(n_clicks, export_format, year_range, selected_platforms, selected_generations, selected_genres, 
               selected_publishers, critic_range, search_value):
    if not n_clicks:
        return None
    
    # Filter data based on user selections
    filtered_df = apply_filters(year_range, selected_platforms, selected_generations, selected_genres, 
                                selected_publishers, critic_range, search_value)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"video_game_sales_{timestamp}"
    
    try:
        if export_format == 'csv':
            return dcc.send_data_frame(filtered_df.to_csv, f"{filename}.csv")
        
        elif export_format == 'excel':
            # Create a BytesIO object to store the Excel file
            output = io.BytesIO()
            
            # Create ExcelWriter object with xlsxwriter engine
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Write the data to Excel
                filtered_df.to_excel(writer, sheet_name='Video Game Sales', index=False)
                
                # Get the xlsxwriter workbook and worksheet objects
                workbook = writer.book
                worksheet = writer.sheets['Video Game Sales']
                
                # Add some formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Write the column headers with the defined format
                for col_num, value in enumerate(filtered_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Set column widths
                worksheet.set_column('A:Z', 15)  # Set width of all columns
                
                # Add a title to the sheet
                title_format = workbook.add_format({
                    'bold': True,
                    'font_size': 14
                })
                worksheet.write('A1', 'Video Game Sales Data', title_format)
                
            # Set the output file
            output.seek(0)
            
            return dcc.send_bytes(output.getvalue(), f"{filename}.xlsx")
        
        elif export_format == 'pdf':
            # Create a BytesIO object to store the PDF
            pdf_buffer = io.BytesIO()
            
            # Create PDF canvas
            pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
            pdf.setTitle(f"Video Game Sales Data - {timestamp}")
            
            # Add title
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(50, 750, "Video Game Sales Dashboard - Data Export")
            pdf.setFont("Helvetica", 10)
            pdf.drawString(50, 735, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Add filter information
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(50, 710, "Applied Filters:")
            pdf.setFont("Helvetica", 10)
            
            y_position = 695
            pdf.drawString(60, y_position, f"Year Range: {year_range[0]} - {year_range[1]}")
            y_position -= 15
            
            pdf.drawString(60, y_position, f"Platforms: {', '.join(selected_platforms) if selected_platforms else 'All'}")
            y_position -= 15
            
            pdf.drawString(60, y_position, f"Genres: {', '.join(selected_genres) if selected_genres else 'All'}")
            y_position -= 15
            
            pdf.drawString(60, y_position, f"Publishers: {', '.join(selected_publishers) if selected_publishers else 'All'}")
            y_position -= 15
            
            # Add table headers
            pdf.setFont("Helvetica-Bold", 10)
            headers = ["Title", "Platform", "Publisher", "Genre", "Year", "Total Sales (M)"]
            header_widths = [180, 60, 90, 70, 40, 80]
            x_position = 50
            y_position -= 30
            
            for i, header in enumerate(headers):
                pdf.drawString(x_position, y_position, header)
                x_position += header_widths[i]
            
            # Add table data (limited to first 30 rows)
            pdf.setFont("Helvetica", 8)
            y_position -= 15
            max_rows = min(30, len(filtered_df))
            
            for i in range(max_rows):
                x_position = 50
                row = filtered_df.iloc[i]
                
                # Truncate title if too long
                title = row['title']
                if len(title) > 25:
                    title = title[:22] + "..."
                
                data = [
                    title,
                    str(row['console']),
                    str(row['publisher']) if len(str(row['publisher'])) < 15 else str(row['publisher'])[:12] + "...",
                    str(row['genre']),
                    str(int(row['release_year'])) if not pd.isna(row['release_year']) else "N/A",
                    f"{row['total_sales']:.1f}" if not pd.isna(row['total_sales']) else "N/A"
                ]
                
                for j, value in enumerate(data):
                    pdf.drawString(x_position, y_position, value)
                    x_position += header_widths[j]
                
                y_position -= 12
                
                # Add a new page if needed
                if y_position < 50:
                    pdf.showPage()
                    pdf.setFont("Helvetica-Bold", 12)
                    pdf.drawString(50, 750, "Video Game Sales Data (continued)")
                    pdf.setFont("Helvetica", 8)
                    y_position = 730
            
            # Add summary statistics
            if y_position < 150:
                pdf.showPage()
                y_position = 750
            
            y_position -= 30
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(50, y_position, "Summary Statistics:")
            pdf.setFont("Helvetica", 10)
            
            y_position -= 20
            total_sales = filtered_df['total_sales'].sum()
            avg_score = filtered_df['critic_score'].mean()
            top_genre = filtered_df.groupby('genre')['total_sales'].sum().idxmax()
            top_platform = filtered_df.groupby('console')['total_sales'].sum().idxmax()
            
            pdf.drawString(60, y_position, f"Total Games: {len(filtered_df)}")
            y_position -= 15
            pdf.drawString(60, y_position, f"Total Sales: {total_sales:.1f} million")
            y_position -= 15
            pdf.drawString(60, y_position, f"Average Critic Score: {avg_score:.1f}/10")
            y_position -= 15
            pdf.drawString(60, y_position, f"Top Genre: {top_genre}")
            y_position -= 15
            pdf.drawString(60, y_position, f"Top Platform: {top_platform}")
            
            # Add footer with page count
            pdf.setFont("Helvetica", 8)
            pdf.drawString(280, 30, "Video Game Sales Dashboard - Page 1")
            
            # Note about data export limit
            if len(filtered_df) > max_rows:
                pdf.setFont("Helvetica-Oblique", 8)
                pdf.drawString(50, 30, f"Note: This PDF contains only the first {max_rows} of {len(filtered_df)} matching records.")
            
            # Save the PDF
            pdf.save()
            
            # Return the PDF file
            pdf_buffer.seek(0)
            return dcc.send_bytes(pdf_buffer.getvalue(), f"{filename}.pdf")
        
        else:
            # Default to CSV if format not recognized
            return dcc.send_data_frame(filtered_df.to_csv, f"{filename}.csv")
            
    except Exception as e:
        logger.error(f"Error exporting data in {export_format} format: {str(e)}")
        # Return a simple CSV with error message
        error_df = pd.DataFrame([{"Error": f"Failed to export data: {str(e)}"}])
        return dcc.send_data_frame(error_df.to_csv, f"export_error_{timestamp}.csv")

# Callback to capture clicks on graphs and store selected game data
@app.callback(
    Output('selected-game-data', 'data'),
    [Input('top-games-bar', 'clickData'),
     Input('critic-score-vs-sales', 'clickData'),
     Input('sales-to-score-ratio', 'clickData')],
    prevent_initial_call=True
)
def capture_selected_game(top_games_click, critic_click, ratio_click):
    ctx = callback_context
    if not ctx.triggered:
        return None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    click_data = None
    
    if trigger_id == 'top-games-bar':
        click_data = top_games_click
    elif trigger_id == 'critic-score-vs-sales':
        click_data = critic_click
    elif trigger_id == 'sales-to-score-ratio':
        click_data = ratio_click
    
    if click_data is None:
        return None
    
    # Extract the game title from the click data
    if trigger_id == 'top-games-bar':
        game_title = click_data['points'][0]['y']
    elif trigger_id == 'critic-score-vs-sales':
        game_title = click_data['points'][0]['hovertext']
    elif trigger_id == 'sales-to-score-ratio':
        game_title = click_data['points'][0]['x']
    else:
        return None
    
    # Look up the full game details
    game_data = df[df['title'] == game_title].iloc[0].to_dict() if len(df[df['title'] == game_title]) > 0 else None
    
    return game_data

# Callback to open modal and display game details
@app.callback(
    [Output('game-details-modal', 'is_open'),
     Output('game-details-content', 'children')],
    [Input('selected-game-data', 'data'),
     Input('close-game-details', 'n_clicks')],
    [State('game-details-modal', 'is_open')],
    prevent_initial_call=True
)
def toggle_modal(game_data, close_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open, []
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'close-game-details':
        return False, []
    
    if trigger_id == 'selected-game-data' and game_data:
        # Create content for modal
        content = [
            html.H3(game_data.get('title', 'N/A'), className="mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P([html.Strong("Platform: "), html.Span(game_data.get('console', 'N/A'))]),
                    html.P([html.Strong("Genre: "), html.Span(game_data.get('genre', 'N/A'))]),
                    html.P([html.Strong("Publisher: "), html.Span(game_data.get('publisher', 'N/A'))]),
                    html.P([html.Strong("Developer: "), html.Span(game_data.get('developer', 'N/A'))]),
                    html.P([html.Strong("Release Date: "), html.Span(str(game_data.get('release_date', 'N/A')))]),
                ], width=6),
                dbc.Col([
                    html.P([html.Strong("Total Sales: "), html.Span(f"{game_data.get('total_sales', 'N/A')} million")]),
                    html.P([html.Strong("Critic Score: "), html.Span(f"{game_data.get('critic_score', 'N/A')}/10")]),
                    html.H5("Regional Sales Breakdown:", className="mt-3"),
                    html.P([html.Strong("North America: "), html.Span(f"{game_data.get('na_sales', 'N/A')} million ({game_data.get('na_percent', 'N/A')}%)")]),
                    html.P([html.Strong("Japan: "), html.Span(f"{game_data.get('jp_sales', 'N/A')} million ({game_data.get('jp_percent', 'N/A')}%)")]),
                    html.P([html.Strong("Europe/Australia: "), html.Span(f"{game_data.get('pal_sales', 'N/A')} million ({game_data.get('pal_percent', 'N/A')}%)")]),
                    html.P([html.Strong("Rest of World: "), html.Span(f"{game_data.get('other_sales', 'N/A')} million ({game_data.get('other_percent', 'N/A')}%)")]),
                ], width=6),
            ]),
            html.Hr(),
            html.Div([
                html.H5("Sales Performance Analysis"),
                html.P(f"Commercial Success Ratio: {game_data.get('sales_per_point', 'N/A')} million sales per review point" 
                       if game_data.get('sales_per_point') else "Commercial Success Ratio: Not available"),
            ], className="mt-3")
        ]
        return True, content
    
    return is_open, []

# Callback to change the theme
@app.callback(
    Output("theme-store", "data"),
    Input("theme-selector", "value"),
)
def update_theme(theme_value):
    return {"current_theme": theme_value}

# Callback to update external stylesheets based on theme selection
app.clientside_callback(
    """
    function(theme_data) {
        const theme = theme_data.current_theme || 'Light';
        
        // Map of theme names to their CDN URLs
        const themeMap = {
            'Light': 'https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/bootstrap/bootstrap.min.css',
            'Dark': 'https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/darkly/bootstrap.min.css',
            'Slate': 'https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/slate/bootstrap.min.css',
            'Superhero': 'https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/superhero/bootstrap.min.css'
        };
        
        // Remove the existing stylesheet
        const links = document.getElementsByTagName('link');
        for (let i = 0; i < links.length; i++) {
            const link = links[i];
            if (link.rel === 'stylesheet' && 
                link.href.includes('cdn.jsdelivr.net/npm/bootswatch')) {
                link.parentNode.removeChild(link);
                break;
            }
        }
        
        // Add the new stylesheet
        const newLink = document.createElement('link');
        newLink.rel = 'stylesheet';
        newLink.href = themeMap[theme];
        document.head.appendChild(newLink);
        
        return window.dash_clientside.no_update;
    }
    """,
    Output("theme-div", "children"),
    Input("theme-store", "data"),
)

# Callback for generating forecasts
@app.callback(
    [Output('sales-forecast-chart', 'figure'),
     Output('genre-forecast-chart', 'figure')],
    [Input('forecast-button', 'n_clicks')],
    [State('year-slider', 'value'),
     State('platform-dropdown', 'value'),
     State('console-gen-dropdown', 'value'),
     State('genre-dropdown', 'value'),
     State('publisher-dropdown', 'value'),
     State('forecast-years', 'value'),
     State('model-type', 'value')],
    prevent_initial_call=True
)
def generate_forecast(n_clicks, year_range, selected_platforms, selected_generations, 
                     selected_genres, selected_publishers, forecast_years, model_type):
    # Filter data based on user selections
    filtered_df = apply_filters(year_range, selected_platforms, selected_generations, selected_genres, 
                                selected_publishers, [0, 10], None)
    
    # Group by year to get yearly sales
    yearly_sales = filtered_df.groupby('release_year')['total_sales'].sum().reset_index()
    yearly_sales = yearly_sales.sort_values('release_year')
    
    # Only proceed with forecasting if we have enough data points
    if len(yearly_sales) < 5:
        # Not enough data for a reliable forecast
        fig_forecast = go.Figure()
        fig_forecast.update_layout(
            title="Insufficient data for forecasting (need at least 5 years of data)",
            xaxis_title="Year",
            yaxis_title="Sales (millions)"
        )
        
        fig_genre_forecast = go.Figure()
        fig_genre_forecast.update_layout(
            title="Insufficient data for genre forecasting",
            xaxis_title="Year",
            yaxis_title="Sales (millions)"
        )
        
        return fig_forecast, fig_genre_forecast
    
    # Get the range of years for prediction
    min_year = yearly_sales['release_year'].min()
    max_year = int(yearly_sales['release_year'].max())  # Convert to integer
    
    # Create X (years) and y (sales) for the model
    X = yearly_sales['release_year'].values.reshape(-1, 1)
    y = yearly_sales['total_sales'].values
    
    # Create forecast years
    future_years = np.array(range(max_year + 1, max_year + forecast_years + 1)).reshape(-1, 1)
    
    # Train the model and make predictions
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions for historical and future years
        historical_pred = model.predict(X)
        future_pred = model.predict(future_years)
    else:  # polynomial
        model = make_pipeline(PolynomialFeatures(3), LinearRegression())
        model.fit(X, y)
        
        # Make predictions for historical and future years
        historical_pred = model.predict(X)
        future_pred = model.predict(future_years)
    
    # Create the forecast figure
    fig_forecast = go.Figure()
    
    # Add actual sales data
    fig_forecast.add_trace(
        go.Scatter(
            x=yearly_sales['release_year'],
            y=yearly_sales['total_sales'],
            mode='markers+lines',
            name='Actual Sales',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add fitted model line
    fig_forecast.add_trace(
        go.Scatter(
            x=yearly_sales['release_year'],
            y=historical_pred,
            mode='lines',
            name='Model Fit',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    # Add forecast
    fig_forecast.add_trace(
        go.Scatter(
            x=future_years.flatten(),
            y=future_pred,
            mode='markers+lines',
            name='Forecast',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        )
    )
    
    # Update layout
    model_name = "Linear Regression" if model_type == 'linear' else "Polynomial Regression"
    fig_forecast.update_layout(
        title=f"Video Game Sales Forecast ({model_name})",
        xaxis_title="Year",
        yaxis_title="Total Sales (millions)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add confidence interval
    # Generate genre-specific forecasts for top genres
    top_genres = filtered_df.groupby('genre')['total_sales'].sum().nlargest(5).index.tolist()
    fig_genre_forecast = go.Figure()
    
    for genre in top_genres:
        # Filter data for this genre
        genre_df = filtered_df[filtered_df['genre'] == genre]
        genre_yearly = genre_df.groupby('release_year')['total_sales'].sum().reset_index()
        
        if len(genre_yearly) < 5:
            continue  # Skip genres with insufficient data points
            
        # Prepare data for the model
        X_genre = genre_yearly['release_year'].values.reshape(-1, 1)
        y_genre = genre_yearly['total_sales'].values
        
        # Train model and make predictions
        if model_type == 'linear':
            genre_model = LinearRegression()
            genre_model.fit(X_genre, y_genre)
        else:  # polynomial
            genre_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
            genre_model.fit(X_genre, y_genre)
        
        # Make future predictions
        genre_future_pred = genre_model.predict(future_years)
        
        # Add trace for this genre
        fig_genre_forecast.add_trace(
            go.Scatter(
                x=future_years.flatten(),
                y=genre_future_pred,
                mode='lines+markers',
                name=genre,
                marker=dict(size=7)
            )
        )
    
    # Update genre forecast layout
    fig_genre_forecast.update_layout(
        title=f"Sales Forecast by Genre ({model_name})",
        xaxis_title="Year",
        yaxis_title="Predicted Sales (millions)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig_forecast, fig_genre_forecast

# Add after the generate_forecast callback

# Callback for seasonal analysis charts
@app.callback(
    [Output('seasonal-sales-chart', 'figure'),
     Output('monthly-sales-heatmap', 'figure'),
     Output('quarterly-genre-distribution', 'figure'),
     Output('seasonal-insights-text', 'children')],
    [Input('year-slider', 'value'),
     Input('platform-dropdown', 'value'),
     Input('console-gen-dropdown', 'value'),
     Input('genre-dropdown', 'value'),
     Input('publisher-dropdown', 'value'),
     Input('critic-score-slider', 'value')],
)
def update_seasonal_analysis(year_range, selected_platforms, selected_generations, 
                            selected_genres, selected_publishers, critic_range):
    # Filter data based on user selections
    filtered_df = apply_filters(year_range, selected_platforms, selected_generations, 
                               selected_genres, selected_publishers, critic_range, None)
    
    # Remove entries with missing release dates
    date_df = filtered_df.dropna(subset=['release_date', 'release_month', 'release_quarter'])
    
    # Create a copy to avoid SettingWithCopyWarning
    date_df = date_df.copy()
    
    # Add month names for better readability
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    date_df['month_name'] = date_df['release_month'].map(month_names)
    
    # Add quarter names
    quarter_names = {1: 'Q1 (Jan-Mar)', 2: 'Q2 (Apr-Jun)', 3: 'Q3 (Jul-Sep)', 4: 'Q4 (Oct-Dec)'}
    date_df['quarter_name'] = date_df['release_quarter'].map(quarter_names)
    
    # 1. Create seasonal sales chart (by quarter and year)
    quarterly_sales = date_df.groupby(['release_year', 'quarter_name', 'release_quarter'])['total_sales'].sum().reset_index()
    
    # Sort by quarter to ensure correct order
    quarterly_sales = quarterly_sales.sort_values(['release_year', 'release_quarter'])
    
    fig_seasonal = px.line(
        quarterly_sales,
        x='release_year',
        y='total_sales',
        color='quarter_name',
        markers=True,
        title='Quarterly Sales Trends Over Time',
        labels={
            'release_year': 'Year',
            'total_sales': 'Total Sales (millions)',
            'quarter_name': 'Quarter'
        },
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    # Customize layout
    fig_seasonal.update_layout(
        xaxis=dict(tickmode='linear'),
        legend=dict(orientation="h", title=None, yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # 2. Create monthly sales heatmap
    monthly_sales = date_df.groupby(['release_year', 'release_month'])['total_sales'].sum().reset_index()
    monthly_sales_pivot = monthly_sales.pivot_table(
        values='total_sales',
        index='release_year',
        columns='release_month',
        fill_value=0
    )
    
    # Create column names with month abbreviations
    monthly_sales_pivot.columns = [month_names[m] for m in monthly_sales_pivot.columns]
    
    # Focus on the most relevant years for better visualization (max 20 years)
    if len(monthly_sales_pivot) > 20:
        # Get the years with the most sales
        yearly_totals = monthly_sales_pivot.sum(axis=1).sort_values(ascending=False)
        top_years = yearly_totals.nlargest(20).index
        monthly_sales_pivot = monthly_sales_pivot.loc[top_years]
    
    # Create heatmap
    fig_monthly = px.imshow(
        monthly_sales_pivot.T,  # Transpose to have months on y-axis
        labels=dict(x="Year", y="Month", color="Sales (millions)"),
        x=monthly_sales_pivot.index,
        y=monthly_sales_pivot.columns,
        color_continuous_scale='Viridis',
        title='Monthly Sales Heatmap'
    )
    
    # Customize layout
    fig_monthly.update_xaxes(side="top")
    fig_monthly.update_layout(
        xaxis={'type': 'category'},
        yaxis={'categoryarray': list(month_names.values()), 'type': 'category'}
    )
    
    # 3. Create quarterly genre distribution chart
    genre_quarterly = date_df.groupby(['genre', 'quarter_name'])['total_sales'].sum().reset_index()
    
    # Use only top genres for clarity
    top_genres = date_df.groupby('genre')['total_sales'].sum().nlargest(8).index
    genre_quarterly = genre_quarterly[genre_quarterly['genre'].isin(top_genres)]
    
    # Sort by quarter for consistent order
    quarter_order = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']
    genre_quarterly['quarter_name'] = pd.Categorical(genre_quarterly['quarter_name'], categories=quarter_order, ordered=True)
    genre_quarterly = genre_quarterly.sort_values('quarter_name')
    
    fig_genre_quarterly = px.bar(
        genre_quarterly,
        x='quarter_name',
        y='total_sales',
        color='genre',
        barmode='group',
        title='Genre Performance by Quarter',
        labels={
            'quarter_name': 'Quarter',
            'total_sales': 'Total Sales (millions)',
            'genre': 'Genre'
        },
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    # 4. Generate seasonal insights text
    total_games = len(date_df)
    
    # Find the most popular release quarter
    quarter_counts = date_df['quarter_name'].value_counts()
    popular_quarter = quarter_counts.idxmax()
    quarter_percentage = (quarter_counts.max() / total_games * 100).round(1)
    
    # Find the most popular release month
    month_counts = date_df['month_name'].value_counts()
    popular_month = month_counts.idxmax()
    month_percentage = (month_counts.max() / total_games * 100).round(1)
    
    # Find the quarter with highest average sales
    quarter_avg_sales = date_df.groupby('quarter_name')['total_sales'].mean()
    best_sales_quarter = quarter_avg_sales.idxmax()
    best_sales_value = quarter_avg_sales.max().round(2)
    
    # Find best selling genre by quarter
    best_genre_by_quarter = {}
    for quarter in quarter_order:
        if quarter in genre_quarterly['quarter_name'].values:
            quarter_data = genre_quarterly[genre_quarterly['quarter_name'] == quarter]
            best_genre = quarter_data.loc[quarter_data['total_sales'].idxmax()]['genre']
            best_genre_by_quarter[quarter] = best_genre
    
    # Create insights text
    insights = [
        html.P([
            "Based on the selected filters, the analysis shows the following seasonal patterns:"
        ]),
        html.Ul([
            html.Li([
                f"Most games ({quarter_percentage}%) were released in ",
                html.Strong(f"{popular_quarter}"), 
                f", with {popular_month} being the most popular month ({month_percentage}%)."
            ]),
            html.Li([
                f"Games released in ",
                html.Strong(f"{best_sales_quarter}"),
                f" had the highest average sales of {best_sales_value} million units."
            ]),
            html.Li([
                "Best performing genres by quarter:"
            ]),
            html.Ul([
                html.Li([f"{quarter}: ", html.Strong(f"{genre}")]) 
                for quarter, genre in best_genre_by_quarter.items()
            ])
        ])
    ]
    
    # Check if there's a holiday season impact
    q4_data = date_df[date_df['quarter_name'] == 'Q4 (Oct-Dec)']
    other_quarters = date_df[date_df['quarter_name'] != 'Q4 (Oct-Dec)']
    
    if len(q4_data) > 0 and len(other_quarters) > 0:
        q4_avg = q4_data['total_sales'].mean()
        other_avg = other_quarters['total_sales'].mean()
        
        if q4_avg > other_avg:
            holiday_impact = ((q4_avg / other_avg) - 1) * 100
            insights.append(html.P([
                "Holiday season effect: Games released in Q4 (Oct-Dec) sold on average ",
                html.Strong(f"{holiday_impact:.1f}%"), 
                " more than games released in other quarters."
            ]))
    
    return fig_seasonal, fig_monthly, fig_genre_quarterly, insights

# Add after the seasonal analysis callback

# Callback to populate the game dropdown for comparison
@app.callback(
    Output('game-comparison-dropdown', 'options'),
    [Input('search-bar', 'value')],
)
def update_game_dropdown(search_term):
    if search_term and len(search_term) >= 3:
        # Search for games that match the search term
        matching_games = df[df['title'].str.contains(search_term, case=False, na=False)]
        
        # Sort by total sales (most popular first)
        matching_games = matching_games.sort_values('total_sales', ascending=False)
        
        # Create dropdown options
        options = [
            {'label': f"{row['title']} ({row['console']}, {int(row['release_year']) if not pd.isna(row['release_year']) else 'Unknown'})", 
             'value': row['title']} 
            for _, row in matching_games.head(50).iterrows()  # Limit to top 50 matches
        ]
        return options
    
    # If no search term or too short, return top games
    top_games = df.sort_values('total_sales', ascending=False).head(30)
    options = [
        {'label': f"{row['title']} ({row['console']}, {int(row['release_year']) if not pd.isna(row['release_year']) else 'Unknown'})", 
         'value': row['title']} 
        for _, row in top_games.iterrows()
    ]
    return options

# Callback to generate game comparison visualizations
@app.callback(
    [Output('comparison-sales-chart', 'figure'),
     Output('comparison-regional-chart', 'figure'),
     Output('comparison-metrics-chart', 'figure'),
     Output('comparison-table', 'children'),
     Output('comparison-message', 'children')],
    [Input('compare-button', 'n_clicks')],
    [State('game-comparison-dropdown', 'value')],
    prevent_initial_call=True
)
def compare_games(n_clicks, selected_games):
    if not selected_games or len(selected_games) < 2:
        # Not enough games selected
        return [
            go.Figure().update_layout(title="Select at least 2 games to compare"),
            go.Figure().update_layout(title="Select at least 2 games to compare"),
            go.Figure().update_layout(title="Select at least 2 games to compare"),
            html.P("Please select at least two games to compare.", className="text-center text-muted"),
            html.P("Please select at least two games to compare using the dropdown above.", className="text-danger")
        ]
    
    # Get data for selected games
    comparison_data = df[df['title'].isin(selected_games)].copy()
    
    if len(comparison_data) < len(selected_games):
        # Some selected games might have duplicate names, inform the user
        return [
            go.Figure().update_layout(title="Could not find all selected games"),
            go.Figure().update_layout(title="Could not find all selected games"),
            go.Figure().update_layout(title="Could not find all selected games"),
            html.P("Some selected games were not found in the database.", className="text-center text-muted"),
            html.P("One or more selected games could not be found. Try selecting different games.", className="text-warning")
        ]
    
    # Limit to 5 games for better visualization
    if len(comparison_data) > 5:
        comparison_data = comparison_data.head(5)
        message = html.P("Note: Comparison is limited to 5 games for better visualization.", className="text-info")
    else:
        message = html.P(f"Comparing {len(comparison_data)} games.", className="text-success")
    
    # 1. Sales comparison bar chart
    fig_sales = px.bar(
        comparison_data,
        x='title',
        y=['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales'],
        title='Sales Comparison by Region',
        labels={'value': 'Sales (millions)', 'title': 'Game', 'variable': 'Region'},
        color_discrete_map={
            'total_sales': 'purple',
            'na_sales': 'blue',
            'jp_sales': 'red',
            'pal_sales': 'green',
            'other_sales': 'orange'
        },
        barmode='group',
        hover_data=['console', 'publisher', 'release_year']
    )
    
    # 2. Regional sales distribution (percentage stacked bar)
    # Calculate percentages
    regional_percentages = comparison_data[['title', 'na_percent', 'jp_percent', 'pal_percent', 'other_percent']].copy()
    
    fig_regional = px.bar(
        regional_percentages,
        x='title',
        y=['na_percent', 'jp_percent', 'pal_percent', 'other_percent'],
        title='Regional Sales Distribution (%)',
        labels={'value': 'Percentage of Total Sales', 'title': 'Game', 'variable': 'Region'},
        color_discrete_map={
            'na_percent': 'blue',
            'jp_percent': 'red',
            'pal_percent': 'green',
            'other_percent': 'orange'
        },
        barmode='stack'
    )
    
    # Update legend labels
    region_names = {
        'na_percent': 'North America',
        'jp_percent': 'Japan',
        'pal_percent': 'Europe/Australia',
        'other_percent': 'Rest of World'
    }
    
    newnames = {'na_percent': 'North America', 'jp_percent': 'Japan', 
               'pal_percent': 'Europe/Australia', 'other_percent': 'Rest of World'}
    fig_regional.for_each_trace(lambda t: t.update(name=newnames[t.name]))
    
    # 3. Spider chart for metrics comparison
    # Normalize metrics for radar chart
    metrics = comparison_data.copy()
    
    # Ensure we have critic scores
    metrics['critic_score'] = metrics['critic_score'].fillna(0)
    
    # Create radar chart
    fig_radar = go.Figure()
    
    # Define metrics to compare
    metrics_to_compare = ['total_sales', 'critic_score', 'sales_per_point']
    
    # Get the max value for each metric for normalization
    max_values = {metric: metrics[metric].max() for metric in metrics_to_compare}
    
    # Add a trace for each game
    for _, game in metrics.iterrows():
        # Normalize values (0-1 scale)
        normalized_values = [game[metric] / max_values[metric] if max_values[metric] > 0 else 0 
                             for metric in metrics_to_compare]
        
        # Add game trace
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized_values + [normalized_values[0]],  # Close the loop
            theta=['Total Sales', 'Critic Score', 'Commercial Efficiency'] + ['Total Sales'],  # Close the loop
            fill='toself',
            name=game['title']
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='Normalized Metrics Comparison',
        showlegend=True
    )
    
    # 4. Create a detailed comparison table
    # Format the table data
    table_data = []
    
    for _, game in comparison_data.iterrows():
        release_date = pd.to_datetime(game['release_date']).strftime('%Y-%m-%d') if not pd.isna(game['release_date']) else 'Unknown'
        
        row_data = {
            'Game': game['title'],
            'Platform': game['console'],
            'Publisher': game['publisher'],
            'Genre': game['genre'],
            'Release Date': release_date,
            'Total Sales': f"{game['total_sales']:.2f}M" if not pd.isna(game['total_sales']) else 'N/A',
            'Critic Score': f"{game['critic_score']:.1f}/10" if not pd.isna(game['critic_score']) else 'N/A',
            'NA Sales': f"{game['na_sales']:.2f}M" if not pd.isna(game['na_sales']) else 'N/A',
            'JP Sales': f"{game['jp_sales']:.2f}M" if not pd.isna(game['jp_sales']) else 'N/A',
            'EU/AU Sales': f"{game['pal_sales']:.2f}M" if not pd.isna(game['pal_sales']) else 'N/A',
            'RoW Sales': f"{game['other_sales']:.2f}M" if not pd.isna(game['other_sales']) else 'N/A',
            'Sales/Point': f"{game['sales_per_point']:.2f}" if not pd.isna(game['sales_per_point']) else 'N/A'
        }
        table_data.append(row_data)
    
    # Create the table
    table = dbc.Table.from_dataframe(
        pd.DataFrame(table_data),
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="mt-3"
    )
    
    comparison_table = html.Div([
        html.H5("Detailed Game Comparison"),
        table
    ])
    
    return fig_sales, fig_regional, fig_radar, comparison_table, message

# Run the app
if __name__ == '__main__':
    print("Starting dashboard server...")
    app.run(debug=True)