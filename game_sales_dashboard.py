import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os

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

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Video Game Sales Dashboard"

# Load the data
df = load_data()

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Video Game Sales Dashboard", className="text-center mb-4"),
            html.P("Explore global video game sales data across platforms, genres, and time periods", 
                   className="text-center")
        ], width=12)
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
            )
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
     Input('display-count', 'value')]
)
def update_graphs(year_range, selected_platforms, selected_generations, selected_genres, 
                  selected_publishers, critic_range, sort_method, display_count):
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

# Run the app
if __name__ == '__main__':
    print("Starting dashboard server...")
    app.run(debug=True)