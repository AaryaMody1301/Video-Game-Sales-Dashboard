"""
Data loading and preprocessing module for the dashboard
"""
import os
import pandas as pd
import numpy as np
import logging
import traceback
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_data():
    """
    Load and preprocess the video game sales data
    
    Returns:
        pandas.DataFrame: The preprocessed dataframe
    """
    try:
        logger.info("Loading video game sales data...")
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Construct the absolute path to the CSV file
        csv_path = os.path.join(script_dir, 'vgchartz-2024.csv')
        logger.info(f"Looking for data file at: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # Clean the data
        df = clean_data(df)
        
        # Add derived features
        df = add_derived_features(df)
        
        logger.info(f"Data loaded successfully. {len(df)} records found.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.debug(traceback.format_exc())
        # Return a minimal DataFrame with sample data to allow the app to start
        columns = ['title', 'console', 'publisher', 'developer', 'genre', 'release_date', 
                  'total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'critic_score']
        return pd.DataFrame(columns=columns)


def clean_data(df):
    """
    Clean the raw dataframe by handling missing values and converting data types
    
    Args:
        df (pandas.DataFrame): The raw dataframe
        
    Returns:
        pandas.DataFrame: The cleaned dataframe
    """
    # Convert sales columns to numeric, replacing empty strings with NaN
    sales_columns = ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales']
    for col in sales_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert release_date to datetime, coercing errors to NaT
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    # Extract release year
    df['release_year'] = df['release_date'].dt.year
    
    # Fill NaN with 'Unknown' for categorical columns
    categorical_cols = ['console', 'genre', 'publisher', 'developer']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Convert critic_score to numeric
    df['critic_score'] = pd.to_numeric(df['critic_score'], errors='coerce')
    
    return df


def add_derived_features(df):
    """
    Add derived features to the dataframe for analysis
    
    Args:
        df (pandas.DataFrame): The cleaned dataframe
        
    Returns:
        pandas.DataFrame: The dataframe with added features
    """
    # Create console generation categories (updated with newer consoles)
    console_generations = {
        # Fifth Generation (1993-2002)
        'PS1': 'Fifth Gen', 'PS': 'Fifth Gen', 'N64': 'Fifth Gen', 'SAT': 'Fifth Gen', 'DC': 'Fifth Gen',
        # Sixth Generation (1998-2009)
        'PS2': 'Sixth Gen', 'XB': 'Sixth Gen', 'GC': 'Sixth Gen', 
        # Seventh Generation (2005-2013)
        'PS3': 'Seventh Gen', 'X360': 'Seventh Gen', 'Wii': 'Seventh Gen',
        # Eighth Generation (2012-2020)
        'PS4': 'Eighth Gen', 'XOne': 'Eighth Gen', 'WiiU': 'Eighth Gen', 'NS': 'Eighth Gen',
        # Ninth Generation (2020-)
        'PS5': 'Ninth Gen', 'XSX': 'Ninth Gen', 'XS': 'Ninth Gen', 'XBSX': 'Ninth Gen',
        # Handhelds
        'GBA': 'Handheld', 'DS': 'Handheld', 'PSP': 'Handheld', '3DS': 'Handheld', 'PSV': 'Handheld',
        'Switch': 'Hybrid',
        'PC': 'PC'
    }
    
    # Apply console generation mapping
    df['console_gen'] = df['console'].map(lambda x: console_generations.get(x, 'Other'))
    
    # Print available consoles to help debug
    logger.info(f"Available consoles in dataset: {df['console'].unique()}")
    
    # Create decade column for timeline analysis
    df['decade'] = df['release_year'].apply(lambda x: f"{int(x/10)*10}s" if not pd.isna(x) else "Unknown")
    
    # Calculate commercial success ratio (total sales per critic score point)
    # Avoid division by zero by adding a small epsilon
    df['sales_per_point'] = df['total_sales'] / (df['critic_score'].replace(0, np.nan) + 1e-10)
    
    # Create publisher tier categories based on number of titles
    publisher_counts = df['publisher'].value_counts()
    top_publishers = publisher_counts[publisher_counts > 20].index.tolist()
    df['publisher_tier'] = df['publisher'].apply(lambda x: x if x in top_publishers else 'Other Publishers')
    
    # Calculate regional sales percentage (safely)
    df['total_sales_safe'] = df['total_sales'].replace(0, np.nan)
    df['na_percent'] = (df['na_sales'] / df['total_sales_safe'] * 100).round(1).fillna(0)
    df['jp_percent'] = (df['jp_sales'] / df['total_sales_safe'] * 100).round(1).fillna(0)
    df['pal_percent'] = (df['pal_sales'] / df['total_sales_safe'] * 100).round(1).fillna(0)
    df['other_percent'] = (df['other_sales'] / df['total_sales_safe'] * 100).round(1).fillna(0)
    df = df.drop('total_sales_safe', axis=1)
    
    # Additional metadata for analysis
    df['release_month'] = df['release_date'].dt.month
    df['release_quarter'] = df['release_date'].dt.quarter
    df['has_critic_score'] = ~df['critic_score'].isna()
    
    return df


def apply_filters(df, df_cache, year_range, selected_platforms, selected_generations, selected_genres, 
                 selected_publishers, critic_range, search_value):
    """
    Apply filters to the dataframe based on user selections
    
    Args:
        df (pandas.DataFrame): The complete dataframe
        df_cache (DataFrameCache): Cache for filtered dataframes
        year_range (list): Min and max years to include
        selected_platforms (list): Platforms to include
        selected_generations (list): Console generations to include
        selected_genres (list): Genres to include
        selected_publishers (list): Publishers to include
        critic_range (list): Min and max critic scores to include
        search_value (str): Search string for game titles
        
    Returns:
        pandas.DataFrame: The filtered dataframe
    """
    # Create a unique key for this filter combination
    filters = (
        tuple(year_range), 
        tuple(selected_platforms) if selected_platforms else None,
        tuple(selected_generations) if selected_generations else None,
        tuple(selected_genres) if selected_genres else None,
        tuple(selected_publishers) if selected_publishers else None,
        tuple(critic_range),
        search_value
    )
    
    # Check if we have a cached result
    cached_df = df_cache.get(filters)
    if cached_df is not None:
        return cached_df
    
    # If not cached, filter the data
    filtered_df = df.copy()
    
    # Year range filter
    if not pd.isna(year_range[0]) and not pd.isna(year_range[1]):
        filtered_df = filtered_df[
            (filtered_df['release_year'] >= year_range[0]) & 
            (filtered_df['release_year'] <= year_range[1])
        ]
    
    # Platform filter
    if selected_platforms:
        filtered_df = filtered_df[filtered_df['console'].isin(selected_platforms)]
    
    # Console generation filter
    if selected_generations:
        filtered_df = filtered_df[filtered_df['console_gen'].isin(selected_generations)]
    
    # Genre filter
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    # Publisher filter
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    
    # Critic score filter
    filtered_df = filtered_df[
        (filtered_df['critic_score'] >= critic_range[0]) & 
        (filtered_df['critic_score'] <= critic_range[1]) |
        (filtered_df['critic_score'].isna())  # Include games with no critic score
    ]
    
    # Search filter
    if search_value:
        filtered_df = filtered_df[filtered_df['title'].str.contains(search_value, case=False, na=False)]
    
    # Cache the result before returning
    df_cache.set(filters, filtered_df)
    
    return filtered_df 