"""Dash application factory for the Video Game Sales Dashboard."""

import logging
import time
from typing import Optional

import dash
import dash_bootstrap_components as dbc

from src.callbacks.register_callbacks import register_all_callbacks
from src.data.data_loader import load_data
from src.layouts.main_layout import create_layout

logger = logging.getLogger(__name__)

THEMES = {
    "Light": dbc.themes.BOOTSTRAP,
    "Dark": dbc.themes.DARKLY,
    "Slate": dbc.themes.SLATE,
    "Superhero": dbc.themes.SUPERHERO,
}

DEFAULT_PLOTLY_CONFIG = {
    "use_custom_templates": True,
    "simple_charts": False,
}


def create_app(
    memory_limit_mb: Optional[int] = None,
    cache_size: int = 20,
    disable_custom_templates: bool = False,
    simple_charts: bool = False,
    use_sample_data: bool = False,
) -> dash.Dash:
    """Create and configure the dashboard application.

    The app uses a single synchronous factory because Dash serves through Flask's
    synchronous WSGI lifecycle. Sample data is only used when explicitly requested.
    """
    plotly_config = DEFAULT_PLOTLY_CONFIG.copy()
    plotly_config["use_custom_templates"] = not disable_custom_templates
    plotly_config["simple_charts"] = simple_charts

    started_at = time.perf_counter()
    df, cache = load_data(
        cache_size=cache_size,
        memory_limit_mb=memory_limit_mb,
        use_sample=use_sample_data,
    )

    app = dash.Dash(
        __name__,
        external_stylesheets=[THEMES["Light"]],
        suppress_callback_exceptions=True,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
        title="Video Game Sales Dashboard",
    )

    app.layout = create_layout(df)
    register_all_callbacks(app, df, cache, plotly_config)

    elapsed = time.perf_counter() - started_at
    source = "sample" if use_sample_data else "production"
    logger.info(
        "Dashboard initialized with %s data (%s rows) in %.2fs",
        source,
        len(df),
        elapsed,
    )
    return app


def main() -> None:
    """Run the dashboard with development defaults."""
    logging.basicConfig(level=logging.INFO)
    create_app().run_server(debug=True)


if __name__ == "__main__":
    main()
