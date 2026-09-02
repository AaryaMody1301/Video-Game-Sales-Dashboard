"""Filter panel component for the dashboard."""

from dash import dcc, html


def create_filter_panel(df):
    """Create the dashboard filter panel from the available dataset values."""
    valid_years = df["release_year"].dropna().astype(int)
    min_year = int(valid_years.min()) if not valid_years.empty else 1980
    max_year = int(valid_years.max()) if not valid_years.empty else 2024

    default_start = max(min_year, 1990)
    default_end = min(max_year, 2020)
    if default_start > default_end:
        default_start, default_end = min_year, max_year

    year_marks = {year: str(year) for year in range(min_year, max_year + 1, 5)}
    year_marks[max_year] = str(max_year)

    publisher_sales = (
        df.groupby("publisher", observed=False)["total_sales"]
        .sum()
        .sort_values(ascending=False)
        .head(50)
    )
    publisher_options = [str(publisher) for publisher in publisher_sales.index]

    return html.Div(
        [
            html.H4("Filters", className="mt-3"),
            html.Label("Year Range:"),
            dcc.RangeSlider(
                id="year-slider",
                min=min_year,
                max=max_year,
                value=[default_start, default_end],
                marks=year_marks,
                step=1,
            ),
            html.Label("Select Platform:", className="mt-3"),
            dcc.Dropdown(
                id="platform-dropdown",
                options=[
                    {"label": platform, "value": platform}
                    for platform in sorted(df["console"].dropna().astype(str).unique())
                ],
                value=[],
                multi=True,
            ),
            html.Label("Select Console Generation:", className="mt-3"),
            dcc.Dropdown(
                id="console-gen-dropdown",
                options=[
                    {"label": generation, "value": generation}
                    for generation in sorted(df["console_gen"].dropna().astype(str).unique())
                ],
                value=[],
                multi=True,
            ),
            html.Label("Select Genre:", className="mt-3"),
            dcc.Dropdown(
                id="genre-dropdown",
                options=[
                    {"label": genre, "value": genre}
                    for genre in sorted(df["genre"].dropna().astype(str).unique())
                ],
                value=[],
                multi=True,
            ),
            html.Label("Select Publisher:", className="mt-3"),
            dcc.Dropdown(
                id="publisher-dropdown",
                options=[
                    {"label": publisher, "value": publisher}
                    for publisher in publisher_options
                ],
                value=[],
                multi=True,
            ),
            html.Label("Critic Score Range:", className="mt-3"),
            dcc.RangeSlider(
                id="critic-score-slider",
                min=0,
                max=10,
                value=[0, 10],
                marks={score: str(score) for score in range(0, 11)},
                step=0.5,
            ),
            html.Hr(),
            html.H5("Advanced Options"),
            html.Label("Sort Method:", className="mt-2"),
            dcc.RadioItems(
                id="sort-method",
                options=[
                    {"label": "Total Sales", "value": "total_sales"},
                    {"label": "Critic Score", "value": "critic_score"},
                    {"label": "Release Year", "value": "release_year"},
                ],
                value="total_sales",
                labelStyle={"display": "block", "margin-bottom": "5px"},
            ),
            html.Label("Display Count:", className="mt-2"),
            dcc.Slider(
                id="display-count",
                min=5,
                max=25,
                value=10,
                marks={count: str(count) for count in [5, 10, 15, 20, 25]},
                step=5,
            ),
            html.Label("Search Game Title:", className="mt-3"),
            dcc.Input(
                id="search-bar",
                type="text",
                placeholder="Enter game title...",
                debounce=True,
            ),
            html.Label("Export Format:", className="mt-3"),
            dcc.Dropdown(
                id="export-format-dropdown",
                options=[
                    {"label": "CSV", "value": "csv"},
                    {"label": "Excel", "value": "excel"},
                ],
                value="csv",
                clearable=False,
            ),
            html.Button(
                "Export Data",
                id="export-button",
                className="mt-3 btn btn-primary",
            ),
            dcc.Download(id="download-dataframe-csv"),
        ]
    )
