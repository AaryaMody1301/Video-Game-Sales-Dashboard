"""
Data loading and preprocessing module for the dashboard
"""
import os
import pandas as pd
import numpy as np
import logging
import traceback
import asyncio
from functools import lru_cache, wraps
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Set, cast
import numba
from numba import jit, prange
import gc
import warnings
from datetime import datetime
from src.utils.cache import DataFrameCache

logger = logging.getLogger(__name__)

# Configure numba for best performance
numba.config.THREADING_LAYER = 'threadsafe'
if hasattr(numba.config, 'NUMBA_NUM_THREADS'):
    numba.config.NUMBA_NUM_THREADS = min(16, os.cpu_count() or 4)

@jit(nopython=True, parallel=True, fastmath=True)
def calculate_sales_percentages(na_sales: np.ndarray, jp_sales: np.ndarray, 
                              pal_sales: np.ndarray, other_sales: np.ndarray, 
                              total_sales: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate regional sales percentages using Numba for performance
    
    Args:
        na_sales: North America sales array
        jp_sales: Japan sales array
        pal_sales: PAL (Europe, etc.) sales array
        other_sales: Other regions sales array
        total_sales: Total sales array
        
    Returns:
        Tuple of arrays with percentage values
    """
    n = len(total_sales)
    na_percent = np.zeros(n, dtype=np.float32)
    jp_percent = np.zeros(n, dtype=np.float32)
    pal_percent = np.zeros(n, dtype=np.float32)
    other_percent = np.zeros(n, dtype=np.float32)
    
    for i in prange(n):
        if total_sales[i] > 0:
            na_percent[i] = (na_sales[i] / total_sales[i]) * 100
            jp_percent[i] = (jp_sales[i] / total_sales[i]) * 100
            pal_percent[i] = (pal_sales[i] / total_sales[i]) * 100
            other_percent[i] = (other_sales[i] / total_sales[i]) * 100
    
    return na_percent, jp_percent, pal_percent, other_percent

@lru_cache(maxsize=4)  # Increased cache size for different variants
def load_data(cache_size: int = 20, memory_limit_mb: Optional[int] = None, use_sample: bool = False) -> Tuple[pd.DataFrame, DataFrameCache]:
    """
    Load and preprocess the video game sales data
    
    Args:
        cache_size: Maximum number of items to store in cache
        memory_limit_mb: Memory limit in MB for cache
        use_sample: Whether to use sample data instead of loading from file
    
    Returns:
        Tuple of (processed DataFrame, DataFrameCache)
    """
    start_time = time.time()
    try:
        # If sample data is requested, skip file loading
        if use_sample:
            logger.info("Using sample data as requested")
            return create_sample_data()
            
        logger.info("Loading video game sales data...")
        # Get the project root using pathlib for better cross-platform compatibility
        script_path = Path(__file__)
        project_root = script_path.parent.parent.parent
        
        # Construct the absolute path to the CSV file
        csv_path = project_root / 'vgchartz-2024.csv'
        logger.info(f"Looking for data file at: {csv_path}")
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # Use optimized CSV reading options with newer pandas capabilities
        df = pd.read_csv(
            csv_path,
            low_memory=False,  # Better handling of mixed data types
            dtype={
                'title': str,
                'console': 'category',  # Use category type for string columns with limited unique values
                'publisher': 'category',
                'developer': 'category',
                'genre': 'category',
                'total_sales': 'float32',  # Use more efficient float32 instead of float64
                'na_sales': 'float32',
                'jp_sales': 'float32',
                'pal_sales': 'float32',
                'other_sales': 'float32',
                'critic_score': 'float32'
            },
            parse_dates=['release_date'],  # Pre-parse dates
            engine='c',  # Use the faster C engine
            na_filter=True,  # Enable NA filtering
            cache_dates=True,  # Cache dates for better performance
            on_bad_lines='warn'  # Log warnings for bad lines instead of failing
        )
        
        logger.info(f"Raw data loaded. Processing {len(df)} records...")
        
        # Initialize cache with memory limit
        df_cache = DataFrameCache(max_size=cache_size, max_memory_mb=memory_limit_mb)
        
        # Process data using separate thread for non-blocking operation
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit cleaning task
            clean_future = executor.submit(clean_data, df)
            # Wait for cleaning to complete
            df = clean_future.result()
            
            # Then submit feature generation task
            feature_future = executor.submit(add_derived_features, df)
            # Wait for features to be added
            df = feature_future.result()
        
        # Ensure index is efficient
        df.index = pd.RangeIndex(len(df))
        
        logger.info(f"Data loaded and processed successfully in {time.time() - start_time:.2f} seconds. {len(df)} records.")
        return df, df_cache
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Return a minimal DataFrame with sample data if requested or error occurs
        if use_sample:
            logger.warning("Returning sample data due to loading error")
            return create_sample_data(), DataFrameCache()
        else:
            # Retry with sample data if regular loading fails
            logger.warning("Retrying with sample data")
            return load_data(use_sample=True)

async def load_data_async(cache_size: int = 20, memory_limit_mb: Optional[int] = None) -> Tuple[pd.DataFrame, DataFrameCache]:
    """
    Asynchronously load and preprocess the video game sales data
    
    Args:
        cache_size: Maximum number of items to store in cache
        memory_limit_mb: Memory limit in MB for cache
    
    Returns:
        Tuple of (processed DataFrame, DataFrameCache)
    """
    # Use the event loop to run the synchronous load_data in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: load_data(cache_size, memory_limit_mb))

def create_sample_data() -> Tuple[pd.DataFrame, DataFrameCache]:
    """
    Create a minimal sample dataset for testing or when data loading fails
    
    Returns:
        A tuple of (sample DataFrame, DataFrameCache) containing video game sales data
    """
    columns = ['title', 'console', 'publisher', 'developer', 'genre', 'release_date', 
              'total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'critic_score', 
              'release_year', 'console_gen', 'decade', 'sales_per_point', 'publisher_tier', 
              'na_percent', 'jp_percent', 'pal_percent', 'other_percent', 'release_month', 
              'release_quarter', 'has_critic_score']
    
    # Use NumPy for faster random data generation
    np.random.seed(42)  # Set seed for reproducibility
    
    # Create 30 sample records (increased from 20)
    sample_size = 30
    
    # Pre-generate arrays for better performance
    years = np.random.randint(2000, 2024, sample_size)
    consoles = np.random.choice(['PS4', 'PS5', 'XOne', 'XSX', 'NS', 'PC'], sample_size)
    genres = np.random.choice(['Action', 'Adventure', 'RPG', 'Sports', 'Strategy', 'Simulation', 'Puzzle'], sample_size)
    publishers = np.random.choice(['Nintendo', 'Sony', 'Microsoft', 'EA', 'Activision', 'Ubisoft', 'Take-Two'], sample_size)
    total_sales = np.random.uniform(0.5, 15.0, sample_size)
    
    # Generate dates more efficiently
    today = datetime.now()
    date_offsets = np.random.randint(-365 * 10, 0, sample_size)  # Last 10 years
    dates = [today + pd.Timedelta(days=int(offset)) for offset in date_offsets]
    
    # Create data more efficiently
    data = []
    for i in range(sample_size):
        region_split = np.random.dirichlet(np.ones(4))  # Random regional distribution
        data.append({
            'title': f'Sample Game {i+1}',
            'console': consoles[i],
            'publisher': publishers[i],
            'developer': publishers[i],  # Developer same as publisher for simplicity
            'genre': genres[i],
            'release_date': dates[i],
            'total_sales': total_sales[i],
            'na_sales': total_sales[i] * region_split[0],
            'jp_sales': total_sales[i] * region_split[1],
            'pal_sales': total_sales[i] * region_split[2],
            'other_sales': total_sales[i] * region_split[3],
            'critic_score': np.random.uniform(60, 95),
            'release_year': years[i],
            'console_gen': 'Eighth Gen' if consoles[i] in ['PS4', 'XOne'] else 'Ninth Gen' if consoles[i] in ['PS5', 'XSX'] else 'Hybrid' if consoles[i] == 'NS' else 'PC',
            'decade': '2000s' if years[i] < 2010 else '2010s' if years[i] < 2020 else '2020s',
            'sales_per_point': total_sales[i] / max(85, np.random.uniform(60, 95)),
            'publisher_tier': 'AAA' if publishers[i] in ['Nintendo', 'Sony', 'Microsoft', 'EA', 'Activision'] else 'AA',
            'na_percent': region_split[0] * 100,
            'jp_percent': region_split[1] * 100,
            'pal_percent': region_split[2] * 100,
            'other_percent': region_split[3] * 100,
            'release_month': dates[i].month,
            'release_quarter': (dates[i].month - 1) // 3 + 1,
            'has_critic_score': np.random.choice([True, False], p=[0.8, 0.2])
        })
    
    # Create DataFrame with optimized dtypes
    df = pd.DataFrame(data)
    
    # Convert appropriate columns to categorical
    for col in ['console', 'publisher', 'developer', 'genre', 'console_gen', 'decade', 'publisher_tier']:
        df[col] = df[col].astype('category')
    
    # Convert numeric columns to efficient dtypes
    for col in ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 
                'critic_score', 'sales_per_point', 'na_percent', 'jp_percent', 
                'pal_percent', 'other_percent']:
        df[col] = df[col].astype('float32')
    
    # Integer columns
    for col in ['release_year', 'release_month', 'release_quarter']:
        df[col] = df[col].astype('int16')
    
    # Boolean columns
    df['has_critic_score'] = df['has_critic_score'].astype('bool')
    
    return df, DataFrameCache()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw data
    
    Args:
        df: The raw DataFrame to clean
    
    Returns:
        The cleaned DataFrame
    """
    start_time = time.time()
    logger.info("Cleaning data...")
    
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Drop duplicates
        initial_len = len(df)
        df = df.drop_duplicates(subset=['title', 'console', 'release_date']).reset_index(drop=True)
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df)} duplicate records")
        
        # Handle missing values
        df['critic_score'] = df['critic_score'].fillna(-1)  # Sentinel value for missing critic scores
        
        # Fix date issues
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        # For missing dates, use a default date (January 1, 2000)
        df['release_date'] = df['release_date'].fillna(pd.Timestamp('2000-01-01'))
        
        # Ensure all sales columns are float
        sales_cols = ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales']
        for col in sales_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')
        
        # Ensure total sales is consistent with regional sales
        # If total_sales is missing but we have regional data, calculate the total
        calc_total = df[['na_sales', 'jp_sales', 'pal_sales', 'other_sales']].sum(axis=1)
        mask = (df['total_sales'] == 0) & (calc_total > 0)
        df.loc[mask, 'total_sales'] = calc_total[mask]
        
        # Validate that no total sales are less than sum of regional sales
        invalid_totals = df[df['total_sales'] < calc_total * 0.99]
        if len(invalid_totals) > 0:
            logger.warning(f"Found {len(invalid_totals)} records where total sales is less than sum of regional sales. Fixing...")
            df.loc[df['total_sales'] < calc_total, 'total_sales'] = calc_total
        
        # Convert string columns to category for memory efficiency
        for col in ['console', 'publisher', 'developer', 'genre']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Clean up publisher/developer names
        if 'publisher' in df.columns:
            # Standardize common publisher names - map variations to standard names
            publisher_mapping = {
                'Electronic Arts': 'EA',
                'EA Games': 'EA',
                'Microsoft Game Studios': 'Microsoft',
                'Microsoft Studios': 'Microsoft',
                'Sony Computer Entertainment': 'Sony',
                'Sony Interactive Entertainment': 'Sony',
                'Nintendo of America': 'Nintendo',
                'Activision Blizzard': 'Activision',
                # Add more mappings as needed
            }
            
            # Apply mapping efficiently using pandas replace
            df['publisher'] = df['publisher'].astype(str).replace(publisher_mapping)
            
            # Convert back to category
            df['publisher'] = df['publisher'].astype('category')
        
        # Do the same for developer if present
        if 'developer' in df.columns:
            # Apply the same standardization to developers
            df['developer'] = df['developer'].astype(str).replace(publisher_mapping)
            df['developer'] = df['developer'].astype('category')
        
        # Drop any rows where all sales values are 0
        zero_sales = (df[sales_cols] == 0).all(axis=1)
        if zero_sales.any():
            logger.info(f"Dropping {zero_sales.sum()} rows with zero sales in all regions")
            df = df[~zero_sales]
        
        logger.info(f"Data cleaning completed in {time.time() - start_time:.2f} seconds")
        return df
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        logger.debug(traceback.format_exc())
        return df  # Return the original data if cleaning fails

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the data for analysis
    
    Args:
        df: The cleaned DataFrame
    
    Returns:
        DataFrame with added features
    """
    start_time = time.time()
    logger.info("Adding derived features...")
    
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Extract year, month, and quarter from release date
        if 'release_date' in df.columns:
            df['release_year'] = df['release_date'].dt.year.astype('int16')
            df['release_month'] = df['release_date'].dt.month.astype('int16')
            df['release_quarter'] = ((df['release_month'] - 1) // 3 + 1).astype('int8')
        
        # Add decade
        if 'release_year' in df.columns:
            df['decade'] = pd.cut(
                df['release_year'],
                bins=[1979, 1989, 1999, 2009, 2019, 2029],
                labels=['1980s', '1990s', '2000s', '2010s', '2020s'],
                right=False
            ).astype('category')
        
        # Determine console generation
        if 'console' in df.columns:
            # Map consoles to generations
            console_gens = {
                # First gen
                'ColecoVision': 'First Gen',
                'Commodore': 'First Gen',
                'Atari': 'First Gen',
                # Second gen (8-bit era)
                'NES': 'Second Gen',
                'Master System': 'Second Gen',
                # Third gen (16-bit era)
                'SNES': 'Third Gen',
                'Genesis': 'Third Gen',
                # Fourth gen (32/64-bit era)
                'PS': 'Fourth Gen',
                'N64': 'Fourth Gen',
                'Saturn': 'Fourth Gen',
                # Fifth gen
                'PS2': 'Fifth Gen',
                'Xbox': 'Fifth Gen',
                'GC': 'Fifth Gen',
                'DC': 'Fifth Gen',
                # Sixth gen
                'PS3': 'Sixth Gen',
                'X360': 'Sixth Gen',
                'Wii': 'Sixth Gen',
                # Seventh gen
                'PS4': 'Seventh Gen',
                'XOne': 'Seventh Gen',
                'WiiU': 'Seventh Gen',
                # Eighth gen
                'PS5': 'Eighth Gen',
                'XSX': 'Eighth Gen',
                'NS': 'Eighth Gen',
                # PC is separate
                'PC': 'PC'
            }
            
            # Apply mapping
            df['console_gen'] = df['console'].map(console_gens).fillna('Other').astype('category')
        
        # Add sales per critic point
        if 'critic_score' in df.columns and 'total_sales' in df.columns:
            # Only calculate for games with a critic score
            mask = df['critic_score'] > 0
            df['sales_per_point'] = 0.0
            if mask.any():
                df.loc[mask, 'sales_per_point'] = (df.loc[mask, 'total_sales'] / df.loc[mask, 'critic_score']).astype('float32')
        
        # Add flag for having critic score
        df['has_critic_score'] = (df['critic_score'] > 0).astype('bool')
        
        # Calculate regional sales percentages using Numba for performance
        if all(col in df.columns for col in ['na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'total_sales']):
            # Prepare numpy arrays for calculation
            na_sales = df['na_sales'].values.astype(np.float32)
            jp_sales = df['jp_sales'].values.astype(np.float32)
            pal_sales = df['pal_sales'].values.astype(np.float32)
            other_sales = df['other_sales'].values.astype(np.float32)
            total_sales = df['total_sales'].values.astype(np.float32)
            
            # Calculate percentages using numba
            na_percent, jp_percent, pal_percent, other_percent = calculate_sales_percentages(
                na_sales, jp_sales, pal_sales, other_sales, total_sales
            )
            
            # Assign results back to DataFrame
            df['na_percent'] = na_percent.astype('float32')
            df['jp_percent'] = jp_percent.astype('float32')
            df['pal_percent'] = pal_percent.astype('float32')
            df['other_percent'] = other_percent.astype('float32')
        
        # Assign publisher tiers based on total sales volume
        if 'publisher' in df.columns:
            # Group by publisher and calculate total sales
            publisher_sales = df.groupby('publisher', observed=False)['total_sales'].sum().sort_values(ascending=False)
            
            # Define tiers using quantiles
            tier_thresholds = {
                'AAA': publisher_sales.quantile(0.75),
                'AA': publisher_sales.quantile(0.5),
                'Indie': publisher_sales.quantile(0.25)
            }
            
            # Assign tiers based on total sales
            def assign_tier(sales):
                if sales >= tier_thresholds['AAA']:
                    return 'AAA'
                elif sales >= tier_thresholds['AA']:
                    return 'AA'
                elif sales >= tier_thresholds['Indie']:
                    return 'Indie'
                else:
                    return 'Small'
            
            # Create a mapping dictionary for efficient application
            publisher_tier_map = {
                publisher: assign_tier(sales) 
                for publisher, sales in publisher_sales.items()
            }
            
            # Apply the mapping
            df['publisher_tier'] = df['publisher'].map(publisher_tier_map).astype('category')
        
        logger.info(f"Derived features added in {time.time() - start_time:.2f} seconds")
        return df
        
    except Exception as e:
        logger.error(f"Error adding derived features: {str(e)}")
        logger.debug(traceback.format_exc())
        return df  # Return the original data if feature addition fails

def apply_filters(df: pd.DataFrame, 
                 df_cache: Any,
                 year_range: Optional[List[int]] = None,
                 selected_platforms: Optional[List[str]] = None, 
                 selected_generations: Optional[List[str]] = None,
                 selected_genres: Optional[List[str]] = None,
                 selected_publishers: Optional[List[str]] = None,
                 critic_range: Optional[List[float]] = None,
                 search_value: Optional[str] = None) -> pd.DataFrame:
    """
    Apply filters to the dataframe
    
    Args:
        df: The complete dataframe
        df_cache: Cache for filtered dataframes
        year_range: Optional range of years to filter by [min, max]
        selected_platforms: Optional list of platform names
        selected_generations: Optional list of console generations
        selected_genres: Optional list of genres
        selected_publishers: Optional list of publishers
        critic_range: Optional range of critic scores [min, max]
        search_value: Optional search term for game title
    
    Returns:
        Filtered dataframe
    """
    # Create a cache key from the filters
    filters = (
        tuple(year_range) if year_range else None,
        tuple(selected_platforms) if selected_platforms else None,
        tuple(selected_generations) if selected_generations else None,
        tuple(selected_genres) if selected_genres else None,
        tuple(selected_publishers) if selected_publishers else None,
        tuple(critic_range) if critic_range else None,
        search_value
    )
    
    # Try to get from cache
    cached_df = df_cache.get(filters)
    if cached_df is not None:
        return cached_df
    
    start_time = time.time()
    logger.debug(f"Applying filters: {filters}")
    
    # Start with a copy of the complete dataframe
    filtered_df = df.copy()
    
    # Apply year range filter
    if year_range and len(year_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['release_year'] >= year_range[0]) & 
            (filtered_df['release_year'] <= year_range[1])
        ]
    
    # Apply platform filter
    if selected_platforms and len(selected_platforms) > 0:
        filtered_df = filtered_df[filtered_df['console'].isin(selected_platforms)]
    
    # Apply generation filter
    if selected_generations and len(selected_generations) > 0:
        filtered_df = filtered_df[filtered_df['console_gen'].isin(selected_generations)]
    
    # Apply genre filter
    if selected_genres and len(selected_genres) > 0:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    # Apply publisher filter
    if selected_publishers and len(selected_publishers) > 0:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    
    # Apply critic score filter
    if critic_range and len(critic_range) == 2:
        # Only filter games that have a critic score
        score_mask = (filtered_df['critic_score'] >= critic_range[0]) & (filtered_df['critic_score'] <= critic_range[1])
        has_score = filtered_df['critic_score'] > 0
        filtered_df = filtered_df[score_mask | ~has_score]
    
    # Apply search filter
    if search_value and search_value.strip():
        search_term = search_value.lower().strip()
        filtered_df = filtered_df[filtered_df['title'].str.lower().str.contains(search_term, na=False)]
    
    # Sort by total sales descending
    filtered_df = filtered_df.sort_values('total_sales', ascending=False).reset_index(drop=True)
    
    # Cache the result
    df_cache.set(filters, filtered_df)
    
    logger.debug(f"Filtering completed in {time.time() - start_time:.2f} seconds. {len(filtered_df)} records remain.")
    return filtered_df 