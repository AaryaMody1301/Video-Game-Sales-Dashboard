"""
Callbacks for generating and updating graphs
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from src.data.data_loader import apply_filters
import logging

logger = logging.getLogger(__name__)

def register_graph_callbacks(app, df, df_cache, plotly_config=None):
    """
    Register callbacks for the graph visualizations
    
    Args:
        app (dash.Dash): The Dash application
        df (pandas.DataFrame): The complete dataframe
        df_cache (DataFrameCache): Cache for filtered dataframes
        plotly_config (dict, optional): Configuration options for Plotly charts
    """
    # Default config if none provided
    if plotly_config is None:
        plotly_config = {"use_custom_templates": True, "simple_charts": False}
        
    use_custom_templates = plotly_config.get("use_custom_templates", True)
    simple_charts = plotly_config.get("simple_charts", False)
    
    @app.callback(
        [Output('sales-by-platform', 'figure'),
         Output('sales-by-genre', 'figure'),
         Output('sales-over-time', 'figure'),
         Output('regional-sales-over-time', 'figure'),
         Output('top-games-bar', 'figure'),
         Output('publisher-market-share', 'figure'),
         Output('critic-score-vs-sales', 'figure'),
         Output('regional-sales-comparison', 'figure'),
         Output('genre-trends-over-time', 'figure'),
         Output('console-generation-comparison', 'figure'),
         Output('publisher-performance', 'figure'),
         Output('sales-to-score-ratio', 'figure')],
        [Input('year-slider', 'value'),
         Input('platform-dropdown', 'value'),
         Input('console-gen-dropdown', 'value'),
         Input('genre-dropdown', 'value'),
         Input('publisher-dropdown', 'value'),
         Input('critic-score-slider', 'value'),
         Input('sort-method', 'value'),
         Input('display-count', 'value'),
         Input('search-bar', 'value')]
    )
    def update_graphs(year_range, selected_platforms, selected_generations, selected_genres, 
                      selected_publishers, critic_range, sort_method, display_count, search_value):
        """
        Update all graph visualizations based on user selections
        
        Returns:
            list: List of figure objects for each graph
        """
        # Filter data based on user selections
        filtered_df = apply_filters(df, df_cache, year_range, selected_platforms, selected_generations, selected_genres, 
                                   selected_publishers, critic_range, search_value)
        
        # Special handling for PS5 data - check if there are PS5 games selected but with no sales data
        ps5_games = filtered_df[filtered_df['console'] == 'PS5']
        ps5_with_sales = ps5_games[~ps5_games['total_sales'].isna()]
        ps5_without_sales = len(ps5_games) > 0 and len(ps5_with_sales) == 0
        
        # Check if specific platforms were selected but no data exists for them
        missing_platforms = []
        platforms_without_sales = []
        if selected_platforms:
            for platform in selected_platforms:
                platform_games = filtered_df[filtered_df['console'] == platform]
                if len(platform_games) == 0:
                    missing_platforms.append(platform)
                    logger.info(f"No data found for selected platform: {platform}")
                elif len(platform_games[~platform_games['total_sales'].isna()]) == 0:
                    platforms_without_sales.append(platform)
                    logger.info(f"Platform {platform} has games but no sales data")
        
        # Sales by platform chart
        # We need to filter out NaN values for the groupby operation
        platform_sales_df = filtered_df.dropna(subset=['total_sales'])
        platform_sales = platform_sales_df.groupby('console', observed=False)['total_sales'].sum().reset_index()
        platform_sales = platform_sales.sort_values('total_sales', ascending=False).head(display_count)
        
        # For PS5 or other platforms with data but no sales, we want to show them in the chart
        if ps5_without_sales or platforms_without_sales:
            # If PS5 has games but no sales, add a row with 0 sales for visualization
            for platform in platforms_without_sales:
                if platform not in platform_sales['console'].values:
                    new_row = pd.DataFrame({'console': [platform], 'total_sales': [0]})
                    platform_sales = pd.concat([platform_sales, new_row], ignore_index=True)
            
            platform_sales = platform_sales.sort_values('total_sales', ascending=False)
        
        # Create platform sales chart with compatibility fixes if needed
        if simple_charts:
            fig_platform = go.Figure(go.Bar(
                x=platform_sales['console'],
                y=platform_sales['total_sales'],
                text=platform_sales['total_sales']
            ))
            fig_platform.update_layout(
                title=f'Total Sales by Platform (Top {len(platform_sales)})',
                xaxis_title='Platform',
                yaxis_title='Total Sales (millions)'
            )
        else:
            # Try with px.bar, but if that fails, fall back to go.Figure
            try:
                # Use px.bar with minimal styling to avoid compatibility issues
                fig_platform = px.bar(
                    platform_sales, 
                    x='console', 
                    y='total_sales',
                    title=f'Total Sales by Platform (Top {len(platform_sales)})',
                    labels={'console': 'Platform', 'total_sales': 'Total Sales (millions)'}
                )
                
                # Only add color scale if not using simple charts
                if not simple_charts and use_custom_templates:
                    fig_platform.update_traces(
                        marker_color=platform_sales['total_sales'],
                        marker_colorscale='Viridis'
                    )
            except ValueError:
                # Fall back to basic go.Figure approach
                fig_platform = go.Figure(go.Bar(
                    x=platform_sales['console'],
                    y=platform_sales['total_sales'],
                    text=platform_sales['total_sales'].round(2),
                    marker_color='steelblue'
                ))
                fig_platform.update_layout(
                    title=f'Total Sales by Platform (Top {len(platform_sales)})',
                    xaxis_title='Platform',
                    yaxis_title='Total Sales (millions)'
                )
        
        # Add annotations for special cases
        annotations = []
        
        # Add annotation if platforms were selected but no data exists
        if missing_platforms:
            missing_platforms_str = ", ".join(missing_platforms)
            annotations.append(
                dict(
                    x=0.5, y=0.9,
                    xref="paper", yref="paper",
                    text=f"No data available for: {missing_platforms_str}",
                    showarrow=False,
                    font=dict(color="red", size=12),
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="red",
                    borderwidth=1,
                    borderpad=4
                )
            )
        
        # Add annotation for platforms with games but no sales data
        if platforms_without_sales:
            no_sales_str = ", ".join(platforms_without_sales)
            annotations.append(
                dict(
                    x=0.5, y=0.8,
                    xref="paper", yref="paper",
                    text=f"No sales data available for: {no_sales_str}",
                    showarrow=False,
                    font=dict(color="orange", size=12),
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="orange",
                    borderwidth=1,
                    borderpad=4
                )
            )
        
        # Add all annotations if any exist
        if annotations:
            fig_platform.update_layout(annotations=annotations)
        
        # Sales by genre chart
        genre_sales_df = filtered_df.dropna(subset=['total_sales'])
        genre_sales = genre_sales_df.groupby('genre', observed=False)['total_sales'].sum().reset_index()
        genre_sales = genre_sales.sort_values('total_sales', ascending=False).head(display_count)
        
        if simple_charts:
            fig_genre = go.Figure(go.Pie(
                labels=genre_sales['genre'],
                values=genre_sales['total_sales'],
                hole=0.3
            ))
            fig_genre.update_layout(title='Sales Distribution by Genre')
        else:
            fig_genre = px.pie(
                genre_sales, 
                values='total_sales', 
                names='genre',
                title='Sales Distribution by Genre',
                hole=0.3
            )
            # Only apply custom colors if not using simple charts
            if use_custom_templates and not simple_charts:
                fig_genre.update_traces(
                    marker=dict(colors=px.colors.qualitative.Set3)
                )
        
        # Sales over time chart
        yearly_sales_df = filtered_df.dropna(subset=['total_sales', 'release_year'])
        yearly_sales = yearly_sales_df.groupby('release_year', observed=False)['total_sales'].sum().reset_index()
        yearly_sales = yearly_sales.sort_values('release_year')
        
        if simple_charts:
            fig_time = go.Figure(go.Scatter(
                x=yearly_sales['release_year'],
                y=yearly_sales['total_sales'],
                mode='lines+markers'
            ))
            fig_time.update_layout(
                title='Video Game Sales Trend Over Time',
                xaxis_title='Year',
                yaxis_title='Total Sales (millions)'
            )
        else:
            fig_time = px.line(
                yearly_sales, 
                x='release_year', 
                y='total_sales',
                title='Video Game Sales Trend Over Time',
                labels={'release_year': 'Year', 'total_sales': 'Total Sales (millions)'},
                markers=True
            )
        
        # Add annotations for platforms without sales data if needed
        if platforms_without_sales and not fig_time.layout.annotations:
            no_sales_str = ", ".join(platforms_without_sales)
            fig_time.add_annotation(
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
        
        # Regional sales over time
        regional_yearly_df = filtered_df.dropna(subset=['na_sales', 'jp_sales', 'pal_sales', 'other_sales'], how='all')
        regional_yearly = regional_yearly_df.groupby('release_year', observed=False)[
            ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']
        ].sum().reset_index()
        
        if simple_charts:
            fig_regional_time = go.Figure()
            for col, color in [('na_sales', 'blue'), ('jp_sales', 'red'), 
                              ('pal_sales', 'green'), ('other_sales', 'orange')]:
                fig_regional_time.add_trace(go.Scatter(
                    x=regional_yearly['release_year'],
                    y=regional_yearly[col],
                    mode='lines',
                    stackgroup='one',
                    name=col.replace('_sales', '').upper(),
                    line=dict(color=color)
                ))
            fig_regional_time.update_layout(
                title='Regional Sales Over Time',
                xaxis_title='Year',
                yaxis_title='Sales (millions)'
            )
        else:
            fig_regional_time = px.area(
                regional_yearly, 
                x='release_year', 
                y=['na_sales', 'jp_sales', 'pal_sales', 'other_sales'],
                title='Regional Sales Over Time',
                labels={
                    'release_year': 'Year', 
                    'value': 'Sales (millions)',
                    'variable': 'Region'
                }
            )
            
            # Apply custom colors if not using simple charts
            if use_custom_templates and not simple_charts:
                fig_regional_time.update_traces(
                    line=dict(width=0.5),
                    selector=dict(type='scatter')
                )
                
                # Apply color map
                colors = {
                    'na_sales': 'blue',
                    'jp_sales': 'red',
                    'pal_sales': 'green',
                    'other_sales': 'orange'
                }
                
                for i, col in enumerate(['na_sales', 'jp_sales', 'pal_sales', 'other_sales']):
                    fig_regional_time.data[i].line.color = colors[col]
                    fig_regional_time.data[i].fillcolor = colors[col]
        
        # Add annotations for platforms without sales data if needed
        if platforms_without_sales and not fig_regional_time.layout.annotations:
            no_sales_str = ", ".join(platforms_without_sales)
            fig_regional_time.add_annotation(
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
        
        # Top games bar chart
        if sort_method == 'total_sales':
            sort_col = 'total_sales'
            sort_label = 'Total Sales'
        elif sort_method == 'critic_score':
            sort_col = 'critic_score'
            sort_label = 'Critic Score'
        else:
            sort_col = 'release_year'
            sort_label = 'Release Year'
        
        top_games = filtered_df.sort_values(sort_col, ascending=False)
        top_games = top_games.dropna(subset=[sort_col]).head(display_count)

        if simple_charts:
            fig_top_games = go.Figure(go.Bar(
                x=top_games['total_sales'],
                y=top_games['title'],
                orientation='h',
                text=top_games['total_sales'],
                hovertemplate='%{y}<br>Sales: %{x}<br>Year: %{customdata[0]}<br>Publisher: %{customdata[1]}<br>Score: %{customdata[2]}<br>Platform: %{customdata[3]}',
                customdata=top_games[['release_year', 'publisher', 'critic_score', 'console']]
            ))
            fig_top_games.update_layout(
                title=f'Top {len(top_games)} Games by {sort_label}',
                xaxis_title='Total Sales (millions)',
                yaxis_title='Game Title'
            )
        else:
            fig_top_games = px.bar(
                top_games,
                x='total_sales',
                y='title',
                orientation='h',
                title=f'Top {len(top_games)} Games by {sort_label}',
                labels={'total_sales': 'Total Sales (millions)', 'title': 'Game Title'},
                text='total_sales',
                hover_data=['release_year', 'publisher', 'critic_score', 'console']
            )
            
            # Only add color by genre if not using simple charts
            if use_custom_templates and not simple_charts:
                fig_top_games.update_traces(
                    marker_color=top_games['genre'].astype('category').cat.codes,
                    marker_colorscale='Viridis'
                )
            
        # Publisher market share
        publisher_sales_df = filtered_df.dropna(subset=['total_sales'])
        publisher_sales = publisher_sales_df.groupby('publisher', observed=False)['total_sales'].sum().reset_index()
        publisher_sales = publisher_sales.sort_values('total_sales', ascending=False).head(display_count)
        
        fig_publisher = px.pie(
            publisher_sales, 
            values='total_sales', 
            names='publisher',
            title=f'Top {len(publisher_sales)} Publishers by Market Share',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        # Add annotations for platforms without sales data if needed
        if platforms_without_sales and not fig_publisher.layout.annotations:
            no_sales_str = ", ".join(platforms_without_sales)
            fig_publisher.add_annotation(
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
        
        # Critic score vs sales
        scatter_df = filtered_df.dropna(subset=['critic_score']).copy()
        # Handle NaN values in total_sales by replacing them with a default size value
        scatter_df['size_value'] = scatter_df['total_sales'].fillna(1)  # Replace NaN with 1 for sizing
        
        fig_critic = px.scatter(
            scatter_df,
            x='critic_score',
            y='total_sales',
            title='Critic Score vs. Total Sales',
            labels={'critic_score': 'Critic Score', 'total_sales': 'Total Sales (millions)'},
            color='genre',
            size='size_value',  # Use the cleaned size column
            hover_name='title',
            opacity=0.7,
            size_max=50,
            hover_data=['release_year', 'publisher', 'console']
        )
        
        # Add trendline
        fig_critic.update_layout(showlegend=False)
        
        # Add annotations for platforms without sales data if needed
        if platforms_without_sales and not fig_critic.layout.annotations:
            no_sales_str = ", ".join(platforms_without_sales)
            fig_critic.add_annotation(
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
        
        # Regional sales comparison
        regional_data = pd.melt(
            filtered_df.fillna({'na_sales': 0, 'jp_sales': 0, 'pal_sales': 0, 'other_sales': 0}),
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
        
        regional_totals = regional_data.groupby('region', observed=False)['sales'].sum().reset_index()
        
        fig_regional = px.bar(
            regional_totals,
            x='region',
            y='sales',
            title='Sales by Region',
            labels={'region': 'Region', 'sales': 'Total Sales (millions)'},
            color='region',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Add annotations for platforms without sales data if needed
        if platforms_without_sales and not fig_regional.layout.annotations:
            no_sales_str = ", ".join(platforms_without_sales)
            fig_regional.add_annotation(
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
        
        # Genre trends over time
        genre_yearly_df = filtered_df.dropna(subset=['total_sales'])
        genre_yearly = genre_yearly_df.groupby(['release_year', 'genre'], observed=False)['total_sales'].sum().reset_index()
        genre_yearly = genre_yearly[~genre_yearly['release_year'].isna()]
        
        # Filter to top genres for clarity
        top_genres = genre_yearly_df.groupby('genre', observed=False)['total_sales'].sum().nlargest(8).index.tolist()
        genre_yearly_filtered = genre_yearly[genre_yearly['genre'].isin(top_genres)]
        
        fig_genre_trends = px.line(
            genre_yearly_filtered,
            x='release_year',
            y='total_sales',
            color='genre',
            title='Genre Popularity Trends Over Time',
            labels={'release_year': 'Year', 'total_sales': 'Total Sales (millions)', 'genre': 'Genre'},
            line_shape='spline',
            render_mode='svg'
        )
        
        fig_genre_trends.update_layout(legend_title_text='Genre')
        
        # Add annotations for platforms without sales data if needed
        if platforms_without_sales and not fig_genre_trends.layout.annotations:
            no_sales_str = ", ".join(platforms_without_sales)
            fig_genre_trends.add_annotation(
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
        
        # Console generation comparison
        gen_sales_df = filtered_df.dropna(subset=['total_sales'])
        gen_sales = gen_sales_df.groupby('console_gen', observed=False)['total_sales'].sum().reset_index()
        gen_sales = gen_sales.sort_values('total_sales', ascending=False)
        
        fig_console_gen = px.bar(
            gen_sales,
            x='console_gen',
            y='total_sales',
            title='Sales by Console Generation',
            labels={'console_gen': 'Console Generation', 'total_sales': 'Total Sales (millions)'},
            color='console_gen',
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        
        # Add average critic score on secondary y-axis
        gen_scores = filtered_df.groupby('console_gen', observed=False)['critic_score'].mean().reset_index()
        gen_scores = gen_scores.sort_values('console_gen')
        
        fig_console_gen.add_trace(
            go.Scatter(
                x=gen_scores['console_gen'],
                y=gen_scores['critic_score'],
                name='Avg. Critic Score',
                mode='lines+markers',
                yaxis='y2',
                line=dict(color='red', width=3),
                marker=dict(size=10)
            )
        )
        
        fig_console_gen.update_layout(
            yaxis2=dict(
                title='Average Critic Score',
                overlaying='y',
                side='right',
                range=[0, 10]
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Add annotations for platforms without sales data if needed
        if platforms_without_sales and not fig_console_gen.layout.annotations:
            no_sales_str = ", ".join(platforms_without_sales)
            fig_console_gen.add_annotation(
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
        
        # Publisher performance analysis
        # Get top publishers by game count
        top_n_publishers = filtered_df['publisher'].value_counts().nlargest(display_count).index.tolist()
        publisher_data = filtered_df[filtered_df['publisher'].isin(top_n_publishers)]
        
        # Calculate metrics for each publisher
        publisher_data_clean = publisher_data.dropna(subset=['total_sales', 'critic_score'], how='all')
        publisher_metrics = publisher_data_clean.groupby('publisher', observed=False).agg({
            'total_sales': 'sum',
            'critic_score': 'mean',
            'title': 'count'
        }).reset_index()
        
        publisher_metrics.rename(columns={'title': 'game_count'}, inplace=True)
        
        fig_publisher_perf = px.scatter(
            publisher_metrics,
            x='critic_score',
            y='total_sales',
            size='game_count',
            color='publisher',
            title='Publisher Performance Analysis',
            labels={
                'critic_score': 'Average Critic Score', 
                'total_sales': 'Total Sales (millions)',
                'game_count': 'Number of Games'
            },
            hover_data=['game_count'],
            size_max=60
        )
        
        fig_publisher_perf.update_layout(showlegend=True)
        
        # Add annotations for platforms without sales data if needed
        if platforms_without_sales and not fig_publisher_perf.layout.annotations:
            no_sales_str = ", ".join(platforms_without_sales)
            fig_publisher_perf.add_annotation(
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
        
        # Sales to score ratio (commercial success per review point)
        # Calculate commercial efficiency
        success_ratio = filtered_df.dropna(subset=['critic_score', 'total_sales']).copy()
        success_ratio['sales_per_point'] = success_ratio['total_sales'] / success_ratio['critic_score']
        
        # Get top games by sales_per_point ratio
        top_efficiency = success_ratio.nlargest(display_count, 'sales_per_point')
        
        fig_ratio = px.bar(
            top_efficiency,
            x='title',
            y='sales_per_point',
            title=f'Top {len(top_efficiency)} Games by Commercial Efficiency (Sales per Review Point)',
            labels={
                'title': 'Game', 
                'sales_per_point': 'Sales per Review Point (millions)',
            },
            color='genre',
            hover_data=['total_sales', 'critic_score', 'release_year', 'publisher']
        )
        
        fig_ratio.update_layout(xaxis={'categoryorder': 'total descending'})
        
        # Add annotations for platforms without sales data if needed
        if platforms_without_sales and not fig_ratio.layout.annotations:
            no_sales_str = ", ".join(platforms_without_sales)
            fig_ratio.add_annotation(
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
        
        return [fig_platform, fig_genre, fig_time, fig_regional_time, 
                fig_top_games, fig_publisher, fig_critic, fig_regional,
                fig_genre_trends, fig_console_gen, fig_publisher_perf, fig_ratio] 