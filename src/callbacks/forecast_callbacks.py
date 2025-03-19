"""
Callbacks for forecasting functionality
"""
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from src.data.data_loader import apply_filters

def register_forecast_callbacks(app, df, df_cache):
    """
    Register callbacks for forecasting functionality
    
    Args:
        app (dash.Dash): The Dash application
        df (pandas.DataFrame): The complete dataframe
        df_cache (DataFrameCache): Cache for filtered dataframes
    """
    @app.callback(
        [Output('sales-forecast-chart', 'figure'),
         Output('genre-forecast-chart', 'figure')],
        [Input('forecast-button', 'n_clicks')],
        [State('year-slider', 'value'),
         State('platform-dropdown', 'value'),
         State('console-gen-dropdown', 'value'),
         State('genre-dropdown', 'value'),
         State('publisher-dropdown', 'value'),
         State('forecast-years', 'value'),
         State('model-type', 'value')],
        prevent_initial_call=True
    )
    def generate_forecast(n_clicks, year_range, selected_platforms, selected_generations, 
                         selected_genres, selected_publishers, forecast_years, model_type):
        """
        Generate sales forecasts based on historical data
        
        Returns:
            tuple: (sales_forecast_figure, genre_forecast_figure)
        """
        # Filter data based on user selections
        filtered_df = apply_filters(df, df_cache, year_range, [], [], [], [], [0, 10], None)
        
        # Time series forecast
        yearly_sales = filtered_df.groupby('release_year')['total_sales'].sum().reset_index()
        yearly_sales = yearly_sales.sort_values('release_year')
        yearly_sales = yearly_sales[~yearly_sales['release_year'].isna()]
        
        if len(yearly_sales) < 5:  # Need enough data for a meaningful forecast
            # Return empty charts if not enough data
            empty_fig = px.line(title="Not enough data for forecast")
            return empty_fig, empty_fig
        
        # Prepare data for prediction
        X = yearly_sales['release_year'].values.reshape(-1, 1)
        y = yearly_sales['total_sales'].values
        
        # Fit model based on selected type
        # Convert last_year to int to avoid TypeError in range()
        last_year = int(yearly_sales['release_year'].max())
        future_years = np.array(range(last_year + 1, last_year + forecast_years + 1)).reshape(-1, 1)
        
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X, y)
            future_sales = model.predict(future_years)
        else:  # Polynomial
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            future_years_poly = poly.transform(future_years)
            future_sales = model.predict(future_years_poly)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'release_year': future_years.flatten(),
            'total_sales': future_sales
        })
        
        # Combine with historical data
        combined_df = pd.concat([yearly_sales, forecast_df], ignore_index=True)
        combined_df['data_type'] = combined_df['release_year'].apply(
            lambda x: 'Historical' if x <= last_year else 'Forecast'
        )
        
        # Create the sales forecast chart
        fig_sales_forecast = px.line(
            combined_df, 
            x='release_year', 
            y='total_sales',
            color='data_type',
            title='Sales Forecast',
            labels={'release_year': 'Year', 'total_sales': 'Total Sales (millions)'},
            markers=True
        )
        
        # Add confidence interval
        if len(yearly_sales) > 5:
            std_dev = yearly_sales['total_sales'].std()
            upper_bound = combined_df[combined_df['data_type'] == 'Forecast']['total_sales'] + 1.96 * std_dev
            lower_bound = combined_df[combined_df['data_type'] == 'Forecast']['total_sales'] - 1.96 * std_dev
            lower_bound = lower_bound.clip(lower=0)  # Ensure no negative sales
            
            fig_sales_forecast.add_trace(
                go.Scatter(
                    x=future_years.flatten(),
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            
            fig_sales_forecast.add_trace(
                go.Scatter(
                    x=future_years.flatten(),
                    y=lower_bound,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 100, 80, 0.2)',
                    name='95% Confidence Interval'
                )
            )
        
        # Genre forecast
        genre_yearly = filtered_df.groupby(['release_year', 'genre'])['total_sales'].sum().reset_index()
        genre_yearly = genre_yearly[~genre_yearly['release_year'].isna()]
        
        # Get top genres for clarity
        top_genres = filtered_df.groupby('genre')['total_sales'].sum().nlargest(5).index.tolist()
        genre_yearly_filtered = genre_yearly[genre_yearly['genre'].isin(top_genres)]
        
        # Create a simple genre forecast
        genre_forecast_df = pd.DataFrame()
        
        for genre in top_genres:
            genre_data = genre_yearly_filtered[genre_yearly_filtered['genre'] == genre]
            
            if len(genre_data) >= 5:  # Need enough data
                X_genre = genre_data['release_year'].values.reshape(-1, 1)
                y_genre = genre_data['total_sales'].values
                
                model = LinearRegression()
                model.fit(X_genre, y_genre)
                
                genre_future_sales = model.predict(future_years)
                
                # Create forecast for this genre
                genre_forecast = pd.DataFrame({
                    'release_year': future_years.flatten(),
                    'genre': genre,
                    'total_sales': genre_future_sales,
                    'data_type': 'Forecast'
                })
                
                # Add to overall genre forecast
                genre_forecast_df = pd.concat([genre_forecast_df, genre_forecast], ignore_index=True)
        
        # Combine with historical data
        genre_historical = genre_yearly_filtered.copy()
        genre_historical['data_type'] = 'Historical'
        genre_combined = pd.concat([genre_historical, genre_forecast_df], ignore_index=True)
        
        # Create genre forecast chart
        fig_genre_forecast = px.line(
            genre_combined,
            x='release_year',
            y='total_sales',
            color='genre',
            line_dash='data_type',
            title='Genre Sales Forecast',
            labels={'release_year': 'Year', 'total_sales': 'Total Sales (millions)'}
        )
        
        return fig_sales_forecast, fig_genre_forecast 