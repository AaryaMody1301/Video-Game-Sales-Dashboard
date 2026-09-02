"""Main responsive layout for the Video Game Sales Dashboard."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from src.components.filters import create_filter_panel
from src.components.modals import create_game_details_modal
from src.components.tabs import create_tab_content


def create_layout(df):
    """Create the responsive dashboard layout."""
    valid_years = df["release_year"].dropna()
    min_year = int(valid_years.min()) if not valid_years.empty else "N/A"
    max_year = int(valid_years.max()) if not valid_years.empty else "N/A"

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1("Video Game Sales Dashboard", className="mb-2"),
                            html.P(
                                "Explore reported lifetime game sales across platforms, genres, publishers, and release years.",
                                className="text-muted mb-0",
                            ),
                        ],
                        xs=12,
                        lg=9,
                        className="mb-3",
                    ),
                    dbc.Col(
                        [
                            html.Label("Theme:", className="form-label"),
                            dcc.Dropdown(
                                id="theme-selector",
                                options=[
                                    {"label": "Light", "value": "Light"},
                                    {"label": "Dark", "value": "Dark"},
                                    {"label": "Slate", "value": "Slate"},
                                    {"label": "Superhero", "value": "Superhero"},
                                ],
                                value="Light",
                                clearable=False,
                                style={"width": "100%"},
                            ),
                        ],
                        xs=12,
                        sm=8,
                        md=5,
                        lg=3,
                        className="mb-3",
                    ),
                ],
                align="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        create_filter_panel(df),
                        xs=12,
                        lg=3,
                        className="mb-4",
                    ),
                    dbc.Col(
                        create_tab_content(),
                        xs=12,
                        lg=9,
                        className="mb-4",
                    ),
                ],
                className="g-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Dataset snapshot", className="card-title"),
                                html.P(f"Records after cleaning: {len(df):,}"),
                                html.P(f"Release-year coverage: {min_year} - {max_year}"),
                                html.P(f"Platforms: {df['console'].nunique(dropna=True):,}"),
                                html.P(f"Genres: {df['genre'].nunique(dropna=True):,}"),
                                html.P(f"Publishers: {df['publisher'].nunique(dropna=True):,}"),
                                html.P(
                                    "Built with Dash, Plotly, pandas, scikit-learn, and statsmodels.",
                                    className="text-muted mb-0",
                                ),
                            ]
                        )
                    ),
                    xs=12,
                )
            ),
            create_game_details_modal(),
            dcc.Store(id="selected-game-data"),
            html.Div(id="theme-div", style={"display": "none"}),
            dcc.Store(id="theme-store", data={"current_theme": "Light"}),
        ],
        fluid=True,
        className="py-3 px-3 px-md-4",
    )
