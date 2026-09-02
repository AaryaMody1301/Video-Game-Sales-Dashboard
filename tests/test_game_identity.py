import pandas as pd

from src.utils.game_identity import (
    add_game_ids,
    match_game_from_chart_point,
    select_games_by_id,
)


def _duplicate_title_frame():
    return pd.DataFrame(
        [
            {
                "title": "Shared Game",
                "console": "PS4",
                "release_date": pd.Timestamp("2020-01-01"),
                "release_year": 2020,
                "total_sales": 5.0,
                "critic_score": 8.0,
                "sales_per_point": 0.625,
            },
            {
                "title": "Shared Game",
                "console": "PS5",
                "release_date": pd.Timestamp("2021-02-01"),
                "release_year": 2021,
                "total_sales": 7.0,
                "critic_score": 9.0,
                "sales_per_point": 7.0 / 9.0,
            },
        ]
    )


def test_game_ids_distinguish_same_title_across_platforms():
    identified = add_game_ids(_duplicate_title_frame())

    assert identified["game_id"].nunique() == 2

    selected = select_games_by_id(identified, [identified.iloc[1]["game_id"]])
    assert len(selected) == 1
    assert selected.iloc[0]["console"] == "PS5"


def test_critic_chart_point_resolves_exact_duplicate_title_record():
    df = _duplicate_title_frame()
    point = {"hovertext": "Shared Game", "x": 9.0, "y": 7.0}

    game = match_game_from_chart_point(df, "critic-score-vs-sales", point)

    assert game is not None
    assert game["console"] == "PS5"
    assert game["release_year"] == 2021


def test_ratio_chart_point_resolves_exact_duplicate_title_record():
    df = _duplicate_title_frame()
    point = {"x": "Shared Game", "y": 7.0 / 9.0}

    game = match_game_from_chart_point(df, "sales-to-score-ratio", point)

    assert game is not None
    assert game["console"] == "PS5"
