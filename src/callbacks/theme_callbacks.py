"""
Callbacks for theme switching
"""
from dash.dependencies import Input, Output

def register_theme_callbacks(app):
    """
    Register callbacks for theme switching
    
    Args:
        app (dash.Dash): The Dash application
    """
    # Callback to change the theme
    @app.callback(
        Output("theme-store", "data"),
        Input("theme-selector", "value"),
    )
    def update_theme(theme_value):
        """
        Update the theme data based on user selection
        
        Args:
            theme_value (str): The selected theme
            
        Returns:
            dict: Updated theme data
        """
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