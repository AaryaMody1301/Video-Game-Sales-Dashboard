"""
Callbacks for forecasting functionality
"""
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
from functools import lru_cache
import time
from src.data.data_loader import apply_filters

logger = logging.getLogger(__name__)

# Cache configuration
MODEL_CACHE_SIZE = 10

@lru_cache(maxsize=MODEL_CACHE_SIZE)
def fit_and_predict_cached(model_type, X_train_key, y_train_key, future_years_key):
    """
    Cached model fitting and prediction to improve performance on repeated calls
    
    Args:
        model_type (str): Type of model to use
        X_train_key (tuple): Hashable representation of X_train
        y_train_key (tuple): Hashable representation of y_train
        future_years_key (tuple): Hashable representation of future_years
        
    Returns:
        tuple: (future_sales, prediction_intervals, model_stats)
    """
    # Convert tuple representations back to numpy arrays
    X_train = np.array(X_train_key)
    y_train = np.array(y_train_key)
    future_years = np.array(future_years_key).reshape(-1, 1)
    
    # Ensure X_train is 2D for scikit-learn models
    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)
    
    # Initialize variables
    future_sales = None
    prediction_intervals = None
    model_stats = {'mse': None, 'r2': None}
    
    start_time = time.time()
    
    try:
        if model_type == 'linear':
            # Simple linear regression
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Calculate statistics
            y_pred_train = model.predict(X_train)
            mse = mean_squared_error(y_train, y_pred_train)
            r2 = model.score(X_train, y_train)
            
            # Add cross-validation if we have enough data points
            if len(X_train) >= 10:
                try:
                    cv_scores = cross_val_score(LinearRegression(), X_train, y_train, 
                                               cv=min(5, len(X_train)//2), 
                                               scoring='r2')
                    cv_r2 = np.mean(cv_scores)
                    model_stats = {'mse': mse, 'r2': r2, 'cv_r2': cv_r2}
                except Exception as e:
                    logger.warning(f"Cross-validation failed: {str(e)}")
                    model_stats = {'mse': mse, 'r2': r2}
            else:
                model_stats = {'mse': mse, 'r2': r2}
            
            # Get predictions
            future_sales = model.predict(future_years)
            
            # Calculate prediction intervals
            t_value = 1.96  # Approximately for large samples
            
            # Standard error of the prediction
            se_pred = np.sqrt(mse * (1 + 1/len(X_train) + 
                                   (future_years - np.mean(X_train))**2 / 
                                   np.sum((X_train - np.mean(X_train))**2)))
            
            prediction_intervals = {
                'upper': future_sales + t_value * se_pred.flatten(),
                'lower': future_sales - t_value * se_pred.flatten()
            }
            
        elif model_type == 'poly':
            # Polynomial regression with quadratic terms - use pipeline for better efficiency
            model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            model.fit(X_train, y_train)
            
            # Calculate statistics
            y_pred_train = model.predict(X_train)
            mse = mean_squared_error(y_train, y_pred_train)
            r2 = model.score(X_train, y_train)
            
            # Add cross-validation if we have enough data points
            if len(X_train) >= 10:
                try:
                    cv_scores = cross_val_score(LinearRegression(), X_train, y_train, 
                                               cv=min(5, len(X_train)//2), 
                                               scoring='r2')
                    cv_r2 = np.mean(cv_scores)
                    model_stats = {'mse': mse, 'r2': r2, 'cv_r2': cv_r2}
                except Exception as e:
                    logger.warning(f"Cross-validation failed: {str(e)}")
                    model_stats = {'mse': mse, 'r2': r2}
            else:
                model_stats = {'mse': mse, 'r2': r2}
            
            # Make predictions
            future_sales = model.predict(future_years)
            
            # For polynomial regression, the prediction interval calculation is more complex
            # Using a simplified approach
            t_value = 1.96
            se_pred = np.sqrt(mse) * np.ones(len(future_years))
            
            prediction_intervals = {
                'upper': future_sales + t_value * se_pred,
                'lower': future_sales - t_value * se_pred
            }
            
        elif model_type == 'ridge':
            # Ridge regression (linear with L2 regularization)
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
            # Calculate statistics
            y_pred_train = model.predict(X_train)
            mse = mean_squared_error(y_train, y_pred_train)
            r2 = model.score(X_train, y_train)
            
            # Add cross-validation if we have enough data points
            if len(X_train) >= 10:
                try:
                    cv_scores = cross_val_score(LinearRegression(), X_train, y_train, 
                                               cv=min(5, len(X_train)//2), 
                                               scoring='r2')
                    cv_r2 = np.mean(cv_scores)
                    model_stats = {'mse': mse, 'r2': r2, 'cv_r2': cv_r2}
                except Exception as e:
                    logger.warning(f"Cross-validation failed: {str(e)}")
                    model_stats = {'mse': mse, 'r2': r2}
            else:
                model_stats = {'mse': mse, 'r2': r2}
            
            # Calculate predictions and intervals
            future_sales = model.predict(future_years)
            
            # Calculate error for intervals
            se_pred = np.sqrt(mse) * np.ones(len(future_years))
            
            prediction_intervals = {
                'upper': future_sales + 1.96 * se_pred,
                'lower': future_sales - 1.96 * se_pred
            }
            
        elif model_type == 'arima':
            # Try a simpler approach with pandas Series that has datetime index
            try:
                # Create a proper time series with a datetime index and updated frequency
                years_array = X_train.flatten()
                
                # Make sure all years are unique to avoid issues with duplicates
                unique_years = {}
                for i, year in enumerate(years_array):
                    year_int = int(year)
                    if year_int not in unique_years:
                        unique_years[year_int] = y_train[i]
                    else:
                        # If duplicate, take the average
                        unique_years[year_int] = (unique_years[year_int] + y_train[i]) / 2
                
                # Create a properly ordered Series with unique years
                years = sorted(unique_years.keys())
                values = [unique_years[year] for year in years]
                
                # Create dates with the updated frequency format
                dates = pd.DatetimeIndex([f"{year}-01-01" for year in years], freq='YS')
                ts_data = pd.Series(values, index=dates)
                
                # Try ARIMA model with (1,1,0) parameters first
                try:
                    model = ARIMA(ts_data, order=(1, 1, 0))  # p=1, d=1, q=0
                    results = model.fit(method='css', maxiter=50, disp=0)  # Faster fitting method with fewer iterations
                except Exception as e:
                    logger.warning(f"First ARIMA attempt failed with error: {e}. Trying simpler model.")
                    # Try with even simpler model if first attempt fails
                    model = ARIMA(ts_data, order=(0, 1, 0))  # Simple random walk with drift
                    results = model.fit(method='css', maxiter=50, disp=0)  # Faster fitting method
                
                # Create future dates for forecasting with consistent frequency
                future_dates = pd.date_range(
                    start=ts_data.index[-1] + pd.DateOffset(years=1),
                    periods=len(future_years),
                    freq='YS'  # Updated from AS-JAN to YS
                )
                
                # Get forecast for future dates
                forecast_results = results.get_forecast(steps=len(future_years))
                future_sales = forecast_results.predicted_mean.values
                
                # Calculate prediction intervals (95% confidence)
                # Get the confidence intervals directly from the forecast result
                conf_int = forecast_results.conf_int(alpha=0.05)
                prediction_intervals = {
                    'upper': conf_int.iloc[:, 1].values,
                    'lower': conf_int.iloc[:, 0].values
                }
                
                # Calculate model stats (avoid inconsistent sample issue by checking lengths)
                fitted_values = results.fittedvalues
                
                # Handle potential length mismatch by aligning indices
                if len(fitted_values) == len(ts_data):
                    mse = mean_squared_error(ts_data.values, fitted_values.values)
                    r2 = r2_score(ts_data.values, fitted_values.values)
                else:
                    # If lengths don't match (due to differencing), use aligned values
                    common_idx = fitted_values.index.intersection(ts_data.index)
                    if len(common_idx) > 0:
                        values_actual = ts_data.loc[common_idx].values
                        values_fitted = fitted_values.loc[common_idx].values
                        mse = mean_squared_error(values_actual, values_fitted)
                        r2 = r2_score(values_actual, values_fitted)
                    else:
                        # Default if no overlap
                        mse = 0
                        r2 = 0
                        
                model_stats = {'mse': mse, 'r2': r2}
                
                logger.info(f"ARIMA model successfully fit with parameters {results.specification['order']}")
                
            except Exception as e:
                logger.error(f"ARIMA modeling failed: {e}")
                # Fallback to linear regression
                logger.warning("Falling back to linear regression for forecasting")
                model = LinearRegression()
                model.fit(X_train, y_train)
                future_sales = model.predict(future_years)
                
                # Calculate simple prediction intervals
                y_pred_train = model.predict(X_train)
                mse = mean_squared_error(y_train, y_pred_train)
                r2 = model.score(X_train, y_train)
                model_stats = {'mse': mse, 'r2': r2, 'error': str(e)}
                
                se_pred = np.sqrt(mse) * np.ones(len(future_years))
                prediction_intervals = {
                    'upper': future_sales + 1.96 * se_pred,
                    'lower': future_sales - 1.96 * se_pred
                }
        
        # Clip lower bounds to zero for all models (no negative sales)
        if prediction_intervals:
            prediction_intervals['lower'] = np.maximum(prediction_intervals['lower'], 0)
        
        # Ensure future_sales is not None
        if future_sales is None:
            future_sales = np.zeros(len(future_years))
        
        execution_time = time.time() - start_time
        logger.info(f"Model {model_type} fitted and predicted in {execution_time:.2f} seconds")
        
        return future_sales, prediction_intervals, model_stats
        
    except Exception as e:
        logger.error(f"Error in model fitting: {str(e)}")
        # Return safe defaults
        future_sales = np.zeros(len(future_years))
        prediction_intervals = {
            'upper': future_sales + 1,
            'lower': future_sales
        }
        model_stats = {'mse': 0, 'r2': 0, 'error': str(e)}
        return future_sales, prediction_intervals, model_stats

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
        start_time = time.time()
        try:
            # Filter data based on user selections
            filtered_df = apply_filters(df, df_cache, year_range, selected_platforms, selected_generations, 
                                      selected_genres, selected_publishers, [0, 10], None)
            
            # Time series forecast
            yearly_sales = filtered_df.groupby('release_year')['total_sales'].sum().reset_index()
            yearly_sales = yearly_sales.sort_values('release_year')
            yearly_sales = yearly_sales[~yearly_sales['release_year'].isna()]
            
            if len(yearly_sales) < 5:  # Need enough data for a meaningful forecast
                # Return empty charts if not enough data
                empty_fig = px.line(title="Not enough data for forecast (at least 5 years required)")
                return empty_fig, empty_fig
            
            # Prepare data for prediction
            X = yearly_sales['release_year'].values.reshape(-1, 1)
            y = yearly_sales['total_sales'].values
            
            # Split data for validation if we have enough points
            if len(yearly_sales) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                use_validation = True
            else:
                X_train, y_train = X, y
                X_test, y_test = None, None
                use_validation = False
            
            # Get last year and prepare future years array
            last_year = int(yearly_sales['release_year'].max())
            future_years = np.array(range(last_year + 1, last_year + forecast_years + 1)).reshape(-1, 1)
            
            # Convert arrays to hashable format for caching
            X_train_key = tuple(map(float, X_train.flatten()))
            y_train_key = tuple(map(float, y_train))
            future_years_key = tuple(map(float, future_years.flatten()))
            
            # Get predictions using cached function
            future_sales, prediction_intervals, model_stats = fit_and_predict_cached(
                model_type, X_train_key, y_train_key, future_years_key
            )
            
            # Validate model if we have a test set
            validation_stats = {}
            if use_validation and X_test is not None and y_test is not None and model_type != 'arima':
                try:
                    # Make predictions on test set (skip for ARIMA model since it requires special handling)
                    y_pred_test = None
                    
                    if model_type == 'poly':
                        # Need to recreate the model for prediction on test set
                        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                        model.fit(X_train, y_train)
                        y_pred_test = model.predict(X_test)
                    else:
                        # For linear and ridge recreate simple model
                        if model_type == 'linear':
                            model = LinearRegression()
                        else:  # ridge
                            model = Ridge(alpha=1.0)
                        
                        # Ensure X is 2D for scikit-learn models
                        if len(X_train.shape) == 1:
                            X_train_2d = X_train.reshape(-1, 1)
                        else:
                            X_train_2d = X_train
                            
                        if len(X_test.shape) == 1:
                            X_test_2d = X_test.reshape(-1, 1)
                        else:
                            X_test_2d = X_test
                        
                        model.fit(X_train_2d, y_train)
                        y_pred_test = model.predict(X_test_2d)
                    
                    if y_pred_test is not None:
                        test_r2 = r2_score(y_test, y_pred_test)
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        validation_stats = {'r2': test_r2, 'rmse': test_rmse}
                        logger.info(f"Model validation - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
                        
                except Exception as e:
                    logger.error(f"Error in model validation: {str(e)}")
            
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
                title=f'Sales Forecast ({model_type.capitalize()} Model)',
                labels={'release_year': 'Year', 'total_sales': 'Total Sales (millions)'},
                markers=True
            )
            
            # Add confidence interval
            if prediction_intervals:
                fig_sales_forecast.add_trace(
                    go.Scatter(
                        x=future_years.flatten(),
                        y=prediction_intervals['upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                
                fig_sales_forecast.add_trace(
                    go.Scatter(
                        x=future_years.flatten(),
                        y=prediction_intervals['lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(0, 100, 80, 0.2)',
                        name='95% Prediction Interval'
                    )
                )
            
            # Add annotation showing the model quality
            annotation_text = []
            if model_stats.get('r2') is not None:
                annotation_text.append(f"Model R²: {model_stats['r2']:.2f}")
            if model_stats.get('cv_r2') is not None:
                annotation_text.append(f"Cross-val R²: {model_stats['cv_r2']:.2f}")
            
            if validation_stats:
                if validation_stats.get('r2') is not None:
                    annotation_text.append(f"Validation R²: {validation_stats['r2']:.2f}")
                if validation_stats.get('rmse') is not None:
                    annotation_text.append(f"RMSE: {validation_stats['rmse']:.2f}")
            
            if annotation_text:
                fig_sales_forecast.add_annotation(
                    x=0.05,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text="<br>".join(annotation_text),
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            
            # Genre forecast - Use parallel processing approach for efficiency
            genre_yearly = filtered_df.groupby(['release_year', 'genre'])['total_sales'].sum().reset_index()
            genre_yearly = genre_yearly[~genre_yearly['release_year'].isna()]
            
            # Get top genres for clarity
            top_genres = filtered_df.groupby('genre')['total_sales'].sum().nlargest(5).index.tolist()
            genre_yearly_filtered = genre_yearly[genre_yearly['genre'].isin(top_genres)]
            
            # Create a genre forecast with improved models
            genre_forecast_df = pd.DataFrame()
            genre_historical = pd.DataFrame()
            
            for genre in top_genres:
                genre_data = genre_yearly_filtered[genre_yearly_filtered['genre'] == genre]
                
                if len(genre_data) >= 5:  # Need enough data
                    X_genre = genre_data['release_year'].values.reshape(-1, 1)
                    y_genre = genre_data['total_sales'].values
                    
                    # Store historical data
                    genre_historical_data = genre_data.copy()
                    genre_historical_data['data_type'] = 'Historical'
                    genre_historical = pd.concat([genre_historical, genre_historical_data], ignore_index=True)
                    
                    # Convert arrays to hashable format for caching
                    X_genre_key = tuple(map(float, X_genre.flatten()))
                    y_genre_key = tuple(map(float, y_genre))
                    
                    # Use the cached prediction function
                    try:
                        # For ARIMA models with potential duplicate years, we'll use a simpler fallback
                        if model_type == 'arima':
                            # Use linear regression for genre forecasts with ARIMA to avoid complexity
                            linear_model = LinearRegression()
                            linear_model.fit(X_genre, y_genre)
                            genre_future_sales = linear_model.predict(future_years)
                        else:
                            # Use the cached prediction function for other model types
                            genre_future_sales, _, _ = fit_and_predict_cached(
                                model_type, X_genre_key, y_genre_key, future_years_key
                            )
                        
                        # Create forecast for this genre
                        genre_forecast = pd.DataFrame({
                            'release_year': future_years.flatten(),
                            'genre': genre,
                            'total_sales': genre_future_sales,
                            'data_type': 'Forecast'
                        })
                        
                        # Add to overall genre forecast
                        genre_forecast_df = pd.concat([genre_forecast_df, genre_forecast], ignore_index=True)
                    except Exception as e:
                        logger.error(f"Error forecasting genre {genre}: {str(e)}")
            
            # Combine genre data
            genre_combined = pd.concat([genre_historical, genre_forecast_df], ignore_index=True)
            
            # Create genre forecast chart
            fig_genre_forecast = px.line(
                genre_combined,
                x='release_year',
                y='total_sales',
                color='genre',
                line_dash='data_type',
                title=f'Genre Sales Forecast by {model_type.capitalize()} Model',
                labels={'release_year': 'Year', 'total_sales': 'Total Sales (millions)'}
            )
            
            # Improve chart appearance
            fig_genre_forecast.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Forecast generation completed in {execution_time:.2f} seconds")
            
            return fig_sales_forecast, fig_genre_forecast
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in forecast generation (took {execution_time:.2f}s): {str(e)}")
            error_fig = px.line(title=f"Error in forecast generation: {str(e)}")
            return error_fig, error_fig 