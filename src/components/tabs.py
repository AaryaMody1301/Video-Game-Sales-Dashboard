"""
Tabs component for organizing visualizations
"""
from dash import dcc, html
import dash_bootstrap_components as dbc

def create_tab_content():
    """
    Create the tabs for organizing dashboard visualizations
    
    Returns:
        dash component: The tabs container
    """
    tabs = dbc.Tabs([
        # Market Overview tab
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
        
        # Time Trends tab
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
        
        # Top Performers tab
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
        
        # Sales Analysis tab
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
        
        # Trends Analysis tab
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
        
        # Publisher Insights tab
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
        
        # Predictive Analytics tab
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
        
        # Seasonal Analysis tab
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
        
        # Game Comparison tab
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
    ])
    
    return tabs 