"""Callbacks for theme switching."""

from dash.dependencies import Input, Output


def register_theme_callbacks(app):
    """Register callbacks for theme switching."""

    @app.callback(
        Output("theme-store", "data"),
        Input("theme-selector", "value"),
    )
    def update_theme(theme_value):
        return {"current_theme": theme_value}

    app.clientside_callback(
        """
        function(theme_data) {
            const theme = theme_data.current_theme || 'Light';

            const themeMap = {
                'Light': 'https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/bootstrap/bootstrap.min.css',
                'Dark': 'https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/darkly/bootstrap.min.css',
                'Slate': 'https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/slate/bootstrap.min.css',
                'Superhero': 'https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/superhero/bootstrap.min.css'
            };

            const links = document.getElementsByTagName('link');
            for (let i = 0; i < links.length; i++) {
                const link = links[i];
                if (
                    link.rel === 'stylesheet' &&
                    link.href.includes('cdn.jsdelivr.net/npm/bootswatch')
                ) {
                    link.parentNode.removeChild(link);
                    break;
                }
            }

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
