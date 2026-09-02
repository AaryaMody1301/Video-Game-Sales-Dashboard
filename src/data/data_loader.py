"""Data loading, cleaning, feature engineering, and filtering utilities."""

import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numba
import numpy as np
import pandas as pd
from numba import jit, prange

from src.utils.cache import DataFrameCache

logger = logging.getLogger(__name__)

SALES_COLUMNS = ["total_sales", "na_sales", "jp_sales", "pal_sales", "other_sales"]
CATEGORY_COLUMNS = ["console", "publisher", "developer", "genre"]

PUBLISHER_MAPPING = {
    "Electronic Arts": "EA",
    "EA Games": "EA",
    "Microsoft Game Studios": "Microsoft",
    "Microsoft Studios": "Microsoft",
    "Sony Computer Entertainment": "Sony",
    "Sony Interactive Entertainment": "Sony",
    "Nintendo of America": "Nintendo",
    "Activision Blizzard": "Activision",
}

# Conventional home-console generations. Ambiguous computer/generic platform labels
# are intentionally left as Other rather than assigned a misleading generation.
CONSOLE_GENERATIONS = {
    "ColecoVision": "Second Gen",
    "NES": "Third Gen",
    "Master System": "Third Gen",
    "SNES": "Fourth Gen",
    "Genesis": "Fourth Gen",
    "PS": "Fifth Gen",
    "N64": "Fifth Gen",
    "Saturn": "Fifth Gen",
    "PS2": "Sixth Gen",
    "Xbox": "Sixth Gen",
    "GC": "Sixth Gen",
    "DC": "Sixth Gen",
    "PS3": "Seventh Gen",
    "X360": "Seventh Gen",
    "Wii": "Seventh Gen",
    "PS4": "Eighth Gen",
    "XOne": "Eighth Gen",
    "WiiU": "Eighth Gen",
    "NS": "Eighth Gen",
    "PS5": "Ninth Gen",
    "XSX": "Ninth Gen",
    "PC": "PC",
}

numba.config.THREADING_LAYER = "threadsafe"
if hasattr(numba.config, "NUMBA_NUM_THREADS"):
    numba.config.NUMBA_NUM_THREADS = min(16, numba.config.NUMBA_NUM_THREADS)


@jit(nopython=True, parallel=True, fastmath=True)
def calculate_sales_percentages(
    na_sales: np.ndarray,
    jp_sales: np.ndarray,
    pal_sales: np.ndarray,
    other_sales: np.ndarray,
    total_sales: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate regional shares of each game's reported total sales."""
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


@lru_cache(maxsize=4)
def load_data(
    cache_size: int = 20,
    memory_limit_mb: Optional[int] = None,
    use_sample: bool = False,
) -> Tuple[pd.DataFrame, DataFrameCache]:
    """Load the repository dataset, or deterministic sample data when requested."""
    if use_sample:
        logger.info("Using sample data as requested")
        return create_sample_data()

    csv_path = Path(__file__).resolve().parents[2] / "vgchartz-2024.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    logger.info("Loading video game sales data from %s", csv_path)
    try:
        df = pd.read_csv(
            csv_path,
            low_memory=False,
            dtype={
                "title": str,
                "console": "category",
                "publisher": "category",
                "developer": "category",
                "genre": "category",
                "total_sales": "float32",
                "na_sales": "float32",
                "jp_sales": "float32",
                "pal_sales": "float32",
                "other_sales": "float32",
                "critic_score": "float32",
            },
            parse_dates=["release_date"],
            engine="c",
            on_bad_lines="warn",
        )
        df = add_derived_features(clean_data(df)).reset_index(drop=True)
    except Exception:
        logger.exception("Unable to load or process the production dataset")
        raise

    return df, DataFrameCache(max_size=cache_size, max_memory_mb=memory_limit_mb)


async def load_data_async(
    cache_size: int = 20,
    memory_limit_mb: Optional[int] = None,
) -> Tuple[pd.DataFrame, DataFrameCache]:
    """Load production data without blocking the running event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: load_data(cache_size, memory_limit_mb),
    )


def create_sample_data() -> Tuple[pd.DataFrame, DataFrameCache]:
    """Create deterministic sample data that follows the production schema."""
    rng = np.random.default_rng(42)
    sample_size = 30

    consoles = rng.choice(["PS4", "PS5", "XOne", "XSX", "NS", "PC"], sample_size)
    genres = rng.choice(
        ["Action", "Adventure", "RPG", "Sports", "Strategy", "Simulation", "Puzzle"],
        sample_size,
    )
    publishers = rng.choice(
        ["Nintendo", "Sony", "Microsoft", "EA", "Activision", "Ubisoft", "Take-Two"],
        sample_size,
    )
    total_sales = rng.uniform(0.5, 15.0, sample_size)
    years = rng.integers(2000, 2025, sample_size)

    data = []
    for i in range(sample_size):
        day_offset = int(rng.integers(0, 365))
        release_date = pd.Timestamp(year=int(years[i]), month=1, day=1) + pd.Timedelta(days=day_offset)
        regional_split = rng.dirichlet(np.ones(4))
        critic_score = float(rng.uniform(6.0, 9.5)) if rng.random() < 0.8 else np.nan

        data.append(
            {
                "title": f"Sample Game {i + 1}",
                "console": consoles[i],
                "publisher": publishers[i],
                "developer": publishers[i],
                "genre": genres[i],
                "release_date": release_date,
                "total_sales": total_sales[i],
                "na_sales": total_sales[i] * regional_split[0],
                "jp_sales": total_sales[i] * regional_split[1],
                "pal_sales": total_sales[i] * regional_split[2],
                "other_sales": total_sales[i] * regional_split[3],
                "critic_score": critic_score,
            }
        )

    df = add_derived_features(clean_data(pd.DataFrame(data))).reset_index(drop=True)
    return df, DataFrameCache()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw values without inventing dates or critic scores."""
    df = df.copy()

    df = df.drop_duplicates(subset=["title", "console", "release_date"]).reset_index(drop=True)

    if "critic_score" in df.columns:
        df["critic_score"] = pd.to_numeric(df["critic_score"], errors="coerce").astype("float32")

    # Unknown dates stay unknown. Imputing 2000-01-01 creates a false spike in
    # year, month, quarter, and forecasting analyses.
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    for column in SALES_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("float32")

    regional_total = df[["na_sales", "jp_sales", "pal_sales", "other_sales"]].sum(axis=1)
    missing_total = (df["total_sales"] <= 0) & (regional_total > 0)
    df.loc[missing_total, "total_sales"] = regional_total[missing_total]

    understated_total = df["total_sales"] < regional_total
    if understated_total.any():
        logger.warning(
            "Correcting %s records where total sales are below regional sales",
            int(understated_total.sum()),
        )
        df.loc[understated_total, "total_sales"] = regional_total[understated_total]

    for column in CATEGORY_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string")

    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].replace(PUBLISHER_MAPPING)
    if "developer" in df.columns:
        df["developer"] = df["developer"].replace(PUBLISHER_MAPPING)

    for column in CATEGORY_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("category")

    zero_sales = (df[SALES_COLUMNS] == 0).all(axis=1)
    if zero_sales.any():
        df = df.loc[~zero_sales].copy()

    return df.reset_index(drop=True)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date, console-generation, score, regional, and publisher features."""
    df = df.copy()

    if "release_date" in df.columns:
        release_date = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_date"] = release_date
        df["release_year"] = release_date.dt.year.astype("Int16")
        df["release_month"] = release_date.dt.month.astype("Int8")
        df["release_quarter"] = release_date.dt.quarter.astype("Int8")

        decade_values = release_date.dt.year.map(
            lambda value: f"{(int(value) // 10) * 10}s" if pd.notna(value) else pd.NA
        )
        df["decade"] = decade_values.astype("category")

    if "console" in df.columns:
        console_values = df["console"].astype("string")
        df["console_gen"] = console_values.map(CONSOLE_GENERATIONS).fillna("Other").astype("category")

    if "critic_score" in df.columns and "total_sales" in df.columns:
        has_score = df["critic_score"].notna() & (df["critic_score"] > 0)
        df["has_critic_score"] = has_score.astype("bool")
        df["sales_per_point"] = np.nan
        df.loc[has_score, "sales_per_point"] = (
            df.loc[has_score, "total_sales"] / df.loc[has_score, "critic_score"]
        )
        df["sales_per_point"] = df["sales_per_point"].astype("float32")

    if all(column in df.columns for column in SALES_COLUMNS):
        na_percent, jp_percent, pal_percent, other_percent = calculate_sales_percentages(
            df["na_sales"].to_numpy(dtype=np.float32),
            df["jp_sales"].to_numpy(dtype=np.float32),
            df["pal_sales"].to_numpy(dtype=np.float32),
            df["other_sales"].to_numpy(dtype=np.float32),
            df["total_sales"].to_numpy(dtype=np.float32),
        )
        df["na_percent"] = na_percent
        df["jp_percent"] = jp_percent
        df["pal_percent"] = pal_percent
        df["other_percent"] = other_percent

    if "publisher" in df.columns and "total_sales" in df.columns:
        publisher_sales = df.groupby("publisher", observed=False)["total_sales"].sum()
        if not publisher_sales.empty:
            thresholds = {
                "AAA": publisher_sales.quantile(0.75),
                "AA": publisher_sales.quantile(0.50),
                "Indie": publisher_sales.quantile(0.25),
            }

            def assign_tier(sales: float) -> str:
                if sales >= thresholds["AAA"]:
                    return "AAA"
                if sales >= thresholds["AA"]:
                    return "AA"
                if sales >= thresholds["Indie"]:
                    return "Indie"
                return "Small"

            tier_map = {
                publisher: assign_tier(float(sales))
                for publisher, sales in publisher_sales.items()
            }
            df["publisher_tier"] = df["publisher"].map(tier_map).astype("category")

    return df


def apply_filters(
    df: pd.DataFrame,
    df_cache: Any,
    year_range: Optional[List[int]] = None,
    selected_platforms: Optional[List[str]] = None,
    selected_generations: Optional[List[str]] = None,
    selected_genres: Optional[List[str]] = None,
    selected_publishers: Optional[List[str]] = None,
    critic_range: Optional[List[float]] = None,
    search_value: Optional[str] = None,
) -> pd.DataFrame:
    """Apply dashboard filters and cache the resulting frame."""
    filters = (
        tuple(year_range) if year_range else None,
        tuple(selected_platforms) if selected_platforms else None,
        tuple(selected_generations) if selected_generations else None,
        tuple(selected_genres) if selected_genres else None,
        tuple(selected_publishers) if selected_publishers else None,
        tuple(critic_range) if critic_range else None,
        search_value,
    )

    cached_df = df_cache.get(filters)
    if cached_df is not None:
        return cached_df

    filtered_df = df.copy()

    if year_range and len(year_range) == 2:
        filtered_df = filtered_df[
            filtered_df["release_year"].between(year_range[0], year_range[1], inclusive="both")
        ]

    if selected_platforms:
        filtered_df = filtered_df[filtered_df["console"].isin(selected_platforms)]
    if selected_generations:
        filtered_df = filtered_df[filtered_df["console_gen"].isin(selected_generations)]
    if selected_genres:
        filtered_df = filtered_df[filtered_df["genre"].isin(selected_genres)]
    if selected_publishers:
        filtered_df = filtered_df[filtered_df["publisher"].isin(selected_publishers)]

    # The full 0-10 range means "no score filter" so unrated games remain visible.
    # Once the user narrows the score range, unrated games must not satisfy it.
    if critic_range and len(critic_range) == 2 and tuple(critic_range) != (0, 10):
        filtered_df = filtered_df[
            filtered_df["critic_score"].notna()
            & filtered_df["critic_score"].between(critic_range[0], critic_range[1], inclusive="both")
        ]

    if search_value and search_value.strip():
        filtered_df = filtered_df[
            filtered_df["title"].str.contains(
                search_value.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    filtered_df = filtered_df.sort_values("total_sales", ascending=False).reset_index(drop=True)
    df_cache.set(filters, filtered_df)
    return filtered_df
