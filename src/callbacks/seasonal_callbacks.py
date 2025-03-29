"""
Callbacks for seasonal analysis functionality
"""
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import html
from src.data.data_loader import apply_filters

def register_seasonal_callbacks(app, df, df_cache, plotly_config=None):
    """
    Register callbacks for seasonal analysis functionality
    
    Args:
        app (dash.Dash): The Dash application
        df (pandas.DataFrame): The complete dataframe
        df_cache (DataFrameCache): Cache for filtered dataframes
        plotly_config (dict, optional): Configuration options for Plotly charts
    """
    # Default config if none provided
    if plotly_config is None:
        plotly_config = {"use_custom_templates": True, "simple_charts": False}
    
    @app.callback(
        [Output('seasonal-sales-chart', 'figure'),
         Output('monthly-sales-heatmap', 'figure'),
         Output('quarterly-genre-distribution', 'figure'),
         Output('seasonal-insights-text', 'children')],
        [Input('year-slider', 'value'),
         Input('platform-dropdown', 'value'),
         Input('console-gen-dropdown', 'value'),
         Input('genre-dropdown', 'value'),
         Input('publisher-dropdown', 'value'),
         Input('critic-score-slider', 'value')],
    )
    def update_seasonal_analysis(year_range, selected_platforms, selected_generations, 
                               selected_genres, selected_publishers, critic_range):
        """
        Update seasonal analysis visualizations
        
        Returns:
            tuple: (seasonal_sales_chart, monthly_heatmap, quarterly_genre_chart, insights_text)
        """
        # Filter data based on user selections
        filtered_df = apply_filters(df, df_cache, year_range, selected_platforms, selected_generations, 
                                   selected_genres, selected_publishers, critic_range, None)
        
        # Check if specific platforms were selected but no data exists for them
        missing_platforms = []
        platforms_without_sales = []
        if selected_platforms:
            for platform in selected_platforms:
                platform_games = filtered_df[filtered_df['console'] == platform]
                if len(platform_games) == 0:
                    missing_platforms.append(platform)
                elif len(platform_games[~platform_games['total_sales'].isna()]) == 0:
                    platforms_without_sales.append(platform)
        
        # For seasonal analysis, we need the release month and sales data
        # Filter to rows that have a valid release month and non-null sales
        seasonal_df = filtered_df[~filtered_df['release_month'].isna()]
        seasonal_df_with_sales = seasonal_df.dropna(subset=['total_sales'])
        
        if len(seasonal_df_with_sales) == 0:
            # Create an empty figure with appropriate messaging
            empty_fig = px.bar(title="No sales data available for seasonal analysis")
            
            # Add annotations for platforms without sales data if needed
            if platforms_without_sales:
                no_sales_str = ", ".join(platforms_without_sales)
                empty_fig.add_annotation(
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    text=f"No sales data available for: {no_sales_str}",
                    showarrow=False,
                    font=dict(color="orange", size=12),
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="orange",
                    borderwidth=1,
                    borderpad=4
                )
            
            # Create insights text with explanation
            empty_text = []
            if missing_platforms:
                empty_text.append(html.P(f"No data found for selected platforms: {', '.join(missing_platforms)}"))
            if platforms_without_sales:
                empty_text.append(html.P(f"No sales data available for platforms: {', '.join(platforms_without_sales)}"))
            
            if not empty_text:
                empty_text = [html.P("No seasonal data available with the current filters.")]
                
            return empty_fig, empty_fig, empty_fig, empty_text
        
        # Monthly sales distribution
        monthly_sales = seasonal_df_with_sales.groupby('release_month', observed=False)['total_sales'].sum().reset_index()
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        monthly_sales['month_name'] = monthly_sales['release_month'].map(month_names)
        monthly_sales = monthly_sales.sort_values('release_month')
        
        # Use a simpler chart configuration to avoid the ValueError
        use_simple_charts = plotly_config.get("simple_charts", False)
        
        if use_simple_charts:
            # Use more basic go.Figure approach
            fig_seasonal_sales = go.Figure(go.Bar(
                x=monthly_sales['month_name'],
                y=monthly_sales['total_sales'],
                text=monthly_sales['total_sales'].round(2),
                marker_color='steelblue'
            ))
            fig_seasonal_sales.update_layout(
                title='Monthly Sales Distribution',
                xaxis_title='Month',
                yaxis_title='Total Sales (millions)'
            )
        else:
            try:
                # Try with px.bar but without color parameter that can cause issues
                fig_seasonal_sales = px.bar(
                    monthly_sales,
                    x='month_name',
                    y='total_sales',
                    title='Monthly Sales Distribution',
                    labels={'month_name': 'Month', 'total_sales': 'Total Sales (millions)'}
                )
            except ValueError:
                # Fallback to simpler version if px.bar fails
                fig_seasonal_sales = go.Figure(go.Bar(
                    x=monthly_sales['month_name'],
                    y=monthly_sales['total_sales'],
                    text=monthly_sales['total_sales'].round(2),
                    marker_color='steelblue'
                ))
                fig_seasonal_sales.update_layout(
                    title='Monthly Sales Distribution',
                    xaxis_title='Month',
                    yaxis_title='Total Sales (millions)'
                )
        
        # Add annotations for platforms without sales data if needed
        if platforms_without_sales:
            no_sales_str = ", ".join(platforms_without_sales)
            fig_seasonal_sales.add_annotation(
                x=0.5, y=0.9,
                xref="paper", yref="paper",
                text=f"No sales data available for: {no_sales_str}",
                showarrow=False,
                font=dict(color="orange", size=12),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="orange",
                borderwidth=1,
                borderpad=4
            )
        
        # Monthly sales heatmap by year
        if 'release_year' in seasonal_df_with_sales.columns:
            # Group by year and month
            year_month_sales = seasonal_df_with_sales.groupby(['release_year', 'release_month'], observed=False)['total_sales'].sum().reset_index()
            
            # Pivot the data for the heatmap
            pivot_data = year_month_sales.pivot(
                index='release_year', 
                columns='release_month', 
                values='total_sales'
            ).fillna(0)
            
            # Create the heatmap
            fig_monthly_heatmap = px.imshow(
                pivot_data,
                labels=dict(x="Month", y="Year", color="Sales (millions)"),
                x=[month_names[m] for m in pivot_data.columns],
                y=pivot_data.index,
                title="Monthly Sales Heatmap by Year",
                color_continuous_scale='Viridis'
            )
            
            # Add annotations for platforms without sales data if needed
            if platforms_without_sales:
                no_sales_str = ", ".join(platforms_without_sales)
                fig_monthly_heatmap.add_annotation(
                    x=0.5, y=0.9,
                    xref="paper", yref="paper",
                    text=f"No sales data available for: {no_sales_str}",
                    showarrow=False,
                    font=dict(color="orange", size=12),
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="orange",
                    borderwidth=1,
                    borderpad=4
                )
        else:
            # Fallback if no year data
            fig_monthly_heatmap = px.bar(
                title="Year data not available for heatmap",
                labels={'x': 'Year', 'y': 'Sales (millions)'}
            )
        
        # Quarterly genre distribution
        seasonal_df = filtered_df.copy()
        # Use .loc to avoid SettingWithCopyWarning
        seasonal_df.loc[:, 'quarter'] = pd.to_datetime(seasonal_df['release_date']).dt.quarter
        quarter_genre = seasonal_df.groupby(['quarter', 'genre'], observed=False)['total_sales'].sum().reset_index()
        
        # Limit to top genres
        top_genres = seasonal_df.groupby('genre', observed=False)['total_sales'].sum().nlargest(5).index.tolist()
        quarter_genre_filtered = quarter_genre[quarter_genre['genre'].isin(top_genres)]
        
        fig_quarter_genre = px.bar(
            quarter_genre_filtered,
            x='quarter',
            y='total_sales',
            color='genre',
            title='Quarterly Sales by Genre',
            labels={'quarter': 'Quarter', 'total_sales': 'Total Sales (millions)', 'genre': 'Genre'},
            barmode='group'
        )
        
        # Add annotations for platforms without sales data if needed
        if platforms_without_sales:
            no_sales_str = ", ".join(platforms_without_sales)
            fig_quarter_genre.add_annotation(
                x=0.5, y=0.9,
                xref="paper", yref="paper",
                text=f"No sales data available for: {no_sales_str}",
                showarrow=False,
                font=dict(color="orange", size=12),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="orange",
                borderwidth=1,
                borderpad=4
            )
        
        # Generate insights text
        insights = []
        
        # Find peak sales month
        peak_month = monthly_sales.loc[monthly_sales['total_sales'].idxmax()]
        insights.append(f"Peak sales month is {peak_month['month_name']} with {peak_month['total_sales']:.1f} million in sales.")
        
        # Compare Q4 (holiday season) to other quarters
        q4_sales = seasonal_df_with_sales[seasonal_df_with_sales['release_quarter'] == 4]['total_sales'].sum()
        total_sales = seasonal_df_with_sales['total_sales'].sum()
        q4_percent = (q4_sales / total_sales * 100) if total_sales > 0 else 0
        insights.append(f"Q4 (holiday season) accounts for {q4_percent:.1f}% of annual sales.")
        
        # Genre seasonality
        if len(top_genres) > 0:
            genre_quarters = {}
            for genre in top_genres:
                genre_df = seasonal_df_with_sales[seasonal_df_with_sales['genre'] == genre]
                if len(genre_df) > 0:
                    peak_quarter = genre_df.groupby('release_quarter', observed=False)['total_sales'].sum().idxmax()
                    genre_quarters[genre] = 'Q' + str(peak_quarter)
            
            genre_insights = [f"{genre}: peak in {quarter}" for genre, quarter in genre_quarters.items()]
            insights.append("Genre peak quarters: " + ", ".join(genre_insights))
        
        # Add note about platforms without sales data
        if platforms_without_sales:
            insights.append(f"Note: Platforms without sales data: {', '.join(platforms_without_sales)}")
        
        # Format insights as HTML
        insights_html = [html.P(insight) for insight in insights]
        
        return fig_seasonal_sales, fig_monthly_heatmap, fig_quarter_genre, insights_html 