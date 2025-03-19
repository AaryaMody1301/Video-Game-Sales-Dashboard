"""
Modal components for the dashboard
"""
from dash import html
import dash_bootstrap_components as dbc

def create_game_details_modal():
    """
    Create the game details modal
    
    Returns:
        dash component: The modal
    """
    modal = dbc.Modal(
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
    
    return modal 