"""Helpers for uniquely identifying games across platforms and releases."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def build_game_id(title, console, release_date) -> str:
    date_value = pd.to_datetime(release_date, errors="coerce")
    date_part = date_value.strftime("%Y-%m-%d") if pd.notna(date_value) else "unknown"
    return f"{title}|||{console}|||{date_part}"


def add_game_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["game_id"] = [
        build_game_id(row.title, row.console, row.release_date)
        for row in result[["title", "console", "release_date"]].itertuples(index=False)
    ]
    return result


def game_label(row) -> str:
    release_year = getattr(row, "release_year", None)
    year_label = "Unknown year" if pd.isna(release_year) else str(int(release_year))
    return f"{row.title} — {row.console} ({year_label})"


def select_games_by_id(df: pd.DataFrame, game_ids: Iterable[str]) -> pd.DataFrame:
    selected_ids = list(game_ids)
    if not selected_ids:
        return add_game_ids(df).iloc[0:0].copy()

    identified = add_game_ids(df)
    selected = identified[identified["game_id"].isin(selected_ids)].copy()
    order = {game_id: index for index, game_id in enumerate(selected_ids)}
    selected["_selection_order"] = selected["game_id"].map(order)
    return selected.sort_values("_selection_order").drop(columns="_selection_order")


def _numeric_match(series: pd.Series, value) -> pd.Series:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return pd.Series(False, index=series.index)

    numeric_series = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.isclose(
            numeric_series.to_numpy(dtype=float),
            numeric_value,
            equal_nan=False,
        ),
        index=series.index,
    )


def _match_top_games(df: pd.DataFrame, point: dict) -> pd.DataFrame:
    candidates = df[df["title"] == point.get("y")]
    if "total_sales" in candidates.columns:
        candidates = candidates[
            _numeric_match(candidates["total_sales"], point.get("x"))
        ]
    return candidates


def _match_critic_scatter(df: pd.DataFrame, point: dict) -> pd.DataFrame:
    candidates = df[df["title"] == point.get("hovertext")]
    if "critic_score" in candidates.columns:
        candidates = candidates[
            _numeric_match(candidates["critic_score"], point.get("x"))
        ]
    if "total_sales" in candidates.columns:
        candidates = candidates[
            _numeric_match(candidates["total_sales"], point.get("y"))
        ]
    return candidates


def _match_efficiency_chart(df: pd.DataFrame, point: dict) -> pd.DataFrame:
    candidates = df[df["title"] == point.get("x")]
    if "sales_per_point" in candidates.columns:
        candidates = candidates[
            _numeric_match(candidates["sales_per_point"], point.get("y"))
        ]
    return candidates


def _filter_candidates_by_customdata(
    candidates: pd.DataFrame,
    customdata,
) -> pd.DataFrame:
    if len(candidates) <= 1 or not customdata or "console" not in candidates.columns:
        return candidates

    candidate_consoles = set(candidates["console"].astype(str))
    console_values = {
        str(value)
        for value in customdata
        if str(value) in candidate_consoles
    }
    if not console_values:
        return candidates
    return candidates[candidates["console"].astype(str).isin(console_values)]


def match_game_from_chart_point(
    df: pd.DataFrame,
    trigger_id: str,
    point: dict,
) -> Optional[dict]:
    """Resolve a clicked chart point to one exact game row when possible."""
    if not point:
        return None

    matchers = {
        "top-games-bar": _match_top_games,
        "critic-score-vs-sales": _match_critic_scatter,
        "sales-to-score-ratio": _match_efficiency_chart,
    }
    matcher = matchers.get(trigger_id)
    if matcher is None:
        return None

    candidates = matcher(df, point)
    if candidates.empty:
        return None

    candidates = _filter_candidates_by_customdata(
        candidates,
        point.get("customdata") or [],
    )
    if candidates.empty:
        return None

    return candidates.iloc[0].to_dict()
