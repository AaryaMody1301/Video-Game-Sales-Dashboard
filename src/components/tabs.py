"""Tab layout for dashboard visualizations."""

from dash import dcc, html
import dash_bootstrap_components as dbc


def _graph_row(graph_id: str):
    return dbc.Row(dbc.Col(dcc.Graph(id=graph_id), xs=12), className="g-3")


def create_tab_content():
    """Create the dashboard's visualization tabs."""
    return dbc.Tabs(
        [
            dbc.Tab(
                [
                    _graph_row("sales-by-platform"),
                    _graph_row("sales-by-genre"),
                ],
                label="Market Overview",
            ),
            dbc.Tab(
                [
                    _graph_row("sales-over-time"),
                    _graph_row("regional-sales-over-time"),
                ],
                label="Release Trends",
            ),
            dbc.Tab(
                [
                    _graph_row("top-games-bar"),
                    _graph_row("publisher-market-share"),
                ],
                label="Top Performers",
            ),
            dbc.Tab(
                [
                    _graph_row("critic-score-vs-sales"),
                    _graph_row("regional-sales-comparison"),
                ],
                label="Sales Analysis",
            ),
            dbc.Tab(
                [
                    _graph_row("genre-trends-over-time"),
                    _graph_row("console-generation-comparison"),
                ],
                label="Trends Analysis",
            ),
            dbc.Tab(
                [
                    _graph_row("publisher-performance"),
                    _graph_row("sales-to-score-ratio"),
                ],
                label="Publisher Insights",
            ),
            dbc.Tab(
                [
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                [
                                    html.H4("Sales Forecast", className="mt-3"),
                                    html.P(
                                        "Explore future release-cohort sales using time-aware statistical models."
                                    ),
                                    html.Label("Forecast Years:"),
                                    dcc.Slider(
                                        id="forecast-years",
                                        min=1,
                                        max=10,
                                        value=5,
                                        marks={i: str(i) for i in range(1, 11)},
                                        step=1,
                                    ),
                                    html.Label("Prediction Model:", className="mt-3"),
                                    dcc.RadioItems(
                                        id="model-type",
                                        options=[
                                            {"label": "Linear Regression", "value": "linear"},
                                            {"label": "Polynomial Regression", "value": "poly"},
                                            {"label": "Ridge Regression", "value": "ridge"},
                                            {"label": "ARIMA (Time Series)", "value": "arima"},
                                        ],
                                        value="linear",
                                        labelStyle={"display": "block", "margin-bottom": "5px"},
                                    ),
                                    dbc.Button(
                                        "Generate Forecast",
                                        id="forecast-button",
                                        color="success",
                                        className="mt-3",
                                    ),
                                ],
                                className="p-3 border rounded",
                            ),
                            xs=12,
                        )
                    ),
                    _graph_row("sales-forecast-chart"),
                    _graph_row("genre-forecast-chart"),
                ],
                label="Predictive Analytics",
            ),
            dbc.Tab(
                [
                    _graph_row("seasonal-sales-chart"),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="monthly-sales-heatmap"), xs=12, xl=6),
                            dbc.Col(dcc.Graph(id="quarterly-genre-distribution"), xs=12, xl=6),
                        ],
                        className="g-3",
                    ),
                    dbc.Row(
                        dbc.Col(
                            [
                                html.H5("Release Season Insights", className="mt-3"),
                                html.Div(
                                    id="seasonal-insights-text",
                                    className="p-3 border rounded",
                                ),
                            ],
                            xs=12,
                        )
                    ),
                ],
                label="Release Season Analysis",
            ),
            dbc.Tab(
                [
                    dbc.Row(
                        dbc.Col(
                            [
                                html.H4("Game Comparison", className="mt-3"),
                                html.P(
                                    "Select games to compare reported sales and review metrics side by side."
                                ),
                                dbc.Card(
                                    dbc.CardBody(
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Search and select games to compare:"
                                                        ),
                                                        dcc.Dropdown(
                                                            id="game-comparison-dropdown",
                                                            options=[],
                                                            value=[],
                                                            multi=True,
                                                            placeholder="Type to search for games...",
                                                        ),
                                                    ],
                                                    xs=12,
                                                    lg=9,
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Compare Games",
                                                        id="compare-button",
                                                        color="primary",
                                                        className="mt-3 mt-lg-4 w-100",
                                                    ),
                                                    xs=12,
                                                    lg=3,
                                                ),
                                            ],
                                            className="g-2",
                                        )
                                    )
                                ),
                                html.Div(
                                    id="comparison-message",
                                    className="mt-3 text-muted",
                                ),
                            ],
                            xs=12,
                        )
                    ),
                    _graph_row("comparison-sales-chart"),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id="comparison-regional-chart"), xs=12, xl=6),
                            dbc.Col(dcc.Graph(id="comparison-metrics-chart"), xs=12, xl=6),
                        ],
                        className="g-3",
                    ),
                    dbc.Row(
                        dbc.Col(html.Div(id="comparison-table", className="mt-3"), xs=12)
                    ),
                ],
                label="Game Comparison",
            ),
        ]
    )
