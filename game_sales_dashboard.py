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
import base64
import io
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load and preprocess the data
def load_data():
    print("Loading video game sales data...")
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the CSV file
    csv_path = os.path.join(script_dir, 'vgchartz-2024.csv')
    print(f"Looking for data file at: {csv_path}")
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
    # Create console generation categories
    console_generations = {
        'PS1': 'Fifth Gen', 'PS': 'Fifth Gen', 'N64': 'Fifth Gen', 'SAT': 'Fifth Gen', 'DC': 'Fifth Gen',
        'PS2': 'Sixth Gen', 'XB': 'Sixth Gen', 'GC': 'Sixth Gen', 
        'PS3': 'Seventh Gen', 'X360': 'Seventh Gen', 'Wii': 'Seventh Gen',
        'PS4': 'Eighth Gen', 'XOne': 'Eighth Gen', 'WiiU': 'Eighth Gen', 'NS': 'Eighth Gen',
        'GBA': 'Handheld', 'DS': 'Handheld', 'PSP': 'Handheld', '3DS': 'Handheld', 'PSV': 'Handheld',
        'PC': 'PC'
    }
    
    # Apply console generation mapping
    df['console_gen'] = df['console'].map(lambda x: console_generations.get(x, 'Other'))
    
    # Create decade column for timeline analysis
    df['decade'] = df['release_year'].apply(lambda x: f"{int(x/10)*10}s" if not pd.isna(x) else "Unknown")
    
    # Calculate commercial success ratio (total sales per critic score point)
    df['sales_per_point'] = df['total_sales'] / df['critic_score']
    
    # Create publisher tier categories based on number of titles
    publisher_counts = df['publisher'].value_counts()
    top_publishers = publisher_counts[publisher_counts > 20].index.tolist()
    df['publisher_tier'] = df['publisher'].apply(lambda x: x if x in top_publishers else 'Other Publishers')
    
    # Calculate regional sales percentage
    df['na_percent'] = (df['na_sales'] / df['total_sales'] * 100).round(1)
    df['jp_percent'] = (df['jp_sales'] / df['total_sales'] * 100).round(1)
    df['pal_percent'] = (df['pal_sales'] / df['total_sales'] * 100).round(1)
    df['other_percent'] = (df['other_sales'] / df['total_sales'] * 100).round(1)
    
    print(f"Data loaded successfully. {len(df)} records found.")
    return df

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
    Input("export-button", "n_clicks"),
    State('year-slider', 'value'),
    State('platform-dropdown', 'value'),
    State('console-gen-dropdown', 'value'),
    State('genre-dropdown', 'value'),
    State('publisher-dropdown', 'value'),
    State('critic-score-slider', 'value'),
    State('search-bar', 'value'),
    prevent_initial_call=True,
)
def export_data(n_clicks, year_range, selected_platforms, selected_generations, selected_genres, 
                selected_publishers, critic_range, search_value):
    # Filter data based on user selections
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
    
    return dcc.send_data_frame(filtered_df.to_csv, f"video_game_sales_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

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
    filtered_df = df.copy()
    
    # Apply filters
    if not pd.isna(year_range[0]) and not pd.isna(year_range[1]):
        filtered_df = filtered_df[
            (filtered_df['release_year'] >= year_range[0]) & 
            (filtered_df['release_year'] <= year_range[1])
        ]
    
    if selected_platforms:
        filtered_df = filtered_df[filtered_df['console'].isin(selected_platforms)]
    
    if selected_generations:
        filtered_df = filtered_df[filtered_df['console_gen'].isin(selected_generations)]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    
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

# Run the app
if __name__ == '__main__':
    print("Starting dashboard server...")
    app.run(debug=True)