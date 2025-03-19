"""
Callbacks for the game details modal
"""
import dash
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

def register_game_details_callbacks(app, df):
    """
    Register callbacks for the game details modal
    
    Args:
        app (dash.Dash): The Dash application
        df (pandas.DataFrame): The complete dataframe
    """
    # Callback to capture clicks on graphs and store selected game data
    @app.callback(
        Output('selected-game-data', 'data'),
        [Input('top-games-bar', 'clickData'),
         Input('critic-score-vs-sales', 'clickData'),
         Input('sales-to-score-ratio', 'clickData')],
        prevent_initial_call=True
    )
    def capture_selected_game(top_games_click, critic_click, ratio_click):
        """
        Capture selected game data from graph clicks
        
        Returns:
            dict: Selected game data or None
        """
        ctx = dash.callback_context
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
        """
        Toggle the game details modal and populate content
        
        Returns:
            tuple: (is_open, content)
        """
        ctx = dash.callback_context
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