"""
Callbacks for game comparison functionality
"""
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import html, dash_table

def register_comparison_callbacks(app, df, plotly_config=None):
    """
    Register callbacks for game comparison functionality
    
    Args:
        app (dash.Dash): The Dash application
        df (pandas.DataFrame): The complete dataframe
        plotly_config (dict, optional): Configuration options for Plotly charts
    """
    # Default config if none provided
    if plotly_config is None:
        plotly_config = {"use_custom_templates": True, "simple_charts": False}
    
    @app.callback(
        Output('game-comparison-dropdown', 'options'),
        [Input('search-bar', 'value')],
    )
    def update_game_dropdown(search_term):
        """
        Update the game comparison dropdown options based on search
        
        Returns:
            list: List of dropdown options
        """
        if not search_term:
            # Return top games by sales when no search term
            top_games = df.nlargest(50, 'total_sales')
            return [{'label': game, 'value': game} for game in top_games['title']]
        
        # Filter games based on search term
        filtered_games = df[df['title'].str.contains(search_term, case=False, na=False)]
        filtered_games = filtered_games.nlargest(20, 'total_sales')  # Limit to 20 games
        
        return [{'label': game, 'value': game} for game in filtered_games['title']]

    @app.callback(
        [Output('comparison-sales-chart', 'figure'),
         Output('comparison-regional-chart', 'figure'),
         Output('comparison-metrics-chart', 'figure'),
         Output('comparison-table', 'children'),
         Output('comparison-message', 'children')],
        [Input('compare-button', 'n_clicks')],
        [State('game-comparison-dropdown', 'value')],
        prevent_initial_call=True
    )
    def compare_games(n_clicks, selected_games):
        """
        Compare selected games across different metrics
        
        Returns:
            tuple: (sales_chart, regional_chart, metrics_chart, comparison_table, message)
        """
        if not selected_games or len(selected_games) < 2:
            return [
                px.bar(title="Select at least two games to compare"),
                px.bar(title="Select at least two games to compare"),
                px.bar(title="Select at least two games to compare"),
                html.Div(),
                html.P("Please select at least two games to compare.", className="text-danger")
            ]
        
        # Get data for selected games
        comparison_df = df[df['title'].isin(selected_games)].copy()
        
        if len(comparison_df) < 2:
            return [
                px.bar(title="Selected games not found in the dataset"),
                px.bar(title="Selected games not found in the dataset"),
                px.bar(title="Selected games not found in the dataset"),
                html.Div(),
                html.P("One or more selected games were not found in the dataset.", className="text-danger")
            ]
        
        # Sort by total sales for consistency
        comparison_df = comparison_df.sort_values('total_sales', ascending=False)
        
        # 1. Total sales comparison chart
        fig_sales = px.bar(
            comparison_df,
            x='title',
            y='total_sales',
            title='Total Sales Comparison',
            labels={'title': 'Game', 'total_sales': 'Total Sales (millions)'},
            color='genre',
            hover_data=['publisher', 'console', 'release_year'],
            text='total_sales'
        )
        
        fig_sales.update_traces(texttemplate='%{text:.1f}M', textposition='outside')
        
        # 2. Regional sales comparison chart
        regional_data = pd.melt(
            comparison_df,
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
        
        fig_regional = px.bar(
            regional_data,
            x='title',
            y='sales',
            color='region',
            title='Regional Sales Breakdown',
            labels={'title': 'Game', 'sales': 'Sales (millions)', 'region': 'Region'},
            barmode='group'
        )
        
        # 3. Metrics comparison chart (spider chart)
        # Normalize values for radar chart
        metrics = comparison_df.copy()
        metrics['normalized_sales'] = metrics['total_sales'] / metrics['total_sales'].max() * 100
        metrics['normalized_critic'] = metrics['critic_score'] / 10 * 100  # Scale from 0-10 to 0-100
        
        # Create percentages for regional breakdowns
        metrics['na_percent'] = metrics['na_sales'] / metrics['total_sales'] * 100
        metrics['jp_percent'] = metrics['jp_sales'] / metrics['total_sales'] * 100
        metrics['pal_percent'] = metrics['pal_sales'] / metrics['total_sales'] * 100
        
        fig_metrics = go.Figure()
        
        for i, game in metrics.iterrows():
            fig_metrics.add_trace(go.Scatterpolar(
                r=[
                    game['normalized_sales'],
                    game['normalized_critic'],
                    game['na_percent'],
                    game['jp_percent'],
                    game['pal_percent']
                ],
                theta=['Total Sales', 'Critic Score', 'NA Market', 'JP Market', 'EU Market'],
                fill='toself',
                name=game['title']
            ))
        
        fig_metrics.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title='Game Metrics Comparison'
        )
        
        # 4. Comparison table with detailed metrics
        table_df = comparison_df[['title', 'console', 'publisher', 'genre', 'release_year', 
                                'total_sales', 'critic_score']].copy()
        
        # Format table values
        table_df['total_sales'] = table_df['total_sales'].round(2).astype(str) + ' M'
        table_df['critic_score'] = table_df['critic_score'].round(1).astype(str) + '/10'
        
        # Create the comparison table
        comparison_table = dash_table.DataTable(
            id='game-metrics-table',
            columns=[{"name": col.replace('_', ' ').title(), "id": col} for col in table_df.columns],
            data=table_df.to_dict('records'),
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
        
        # Success message
        message = html.P(f"Comparing {len(comparison_df)} games.", className="text-success")
        
        return fig_sales, fig_regional, fig_metrics, comparison_table, message 