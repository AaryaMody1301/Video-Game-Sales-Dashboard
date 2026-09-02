import numpy as np
import pandas as pd

from src.data.data_loader import (
    add_derived_features,
    apply_filters,
    clean_data,
    create_sample_data,
)
from src.utils.cache import DataFrameCache


def _raw_game(title, console, release_date, critic_score=8.0, total_sales=2.0):
    return {
        "title": title,
        "console": console,
        "publisher": "Publisher",
        "developer": "Developer",
        "genre": "Action",
        "release_date": release_date,
        "total_sales": total_sales,
        "na_sales": total_sales * 0.5,
        "jp_sales": total_sales * 0.1,
        "pal_sales": total_sales * 0.3,
        "other_sales": total_sales * 0.1,
        "critic_score": critic_score,
    }


def test_missing_release_dates_are_not_imputed_to_year_2000():
    raw = pd.DataFrame([_raw_game("Unknown Date", "PS4", None)])

    cleaned = clean_data(raw)
    enriched = add_derived_features(cleaned)

    assert pd.isna(enriched.loc[0, "release_date"])
    assert pd.isna(enriched.loc[0, "release_year"])
    assert pd.isna(enriched.loc[0, "release_month"])
    assert pd.isna(enriched.loc[0, "release_quarter"])


def test_console_generations_match_standard_generations():
    raw = pd.DataFrame(
        [
            _raw_game("PS4 Game", "PS4", "2018-01-01"),
            _raw_game("PS5 Game", "PS5", "2022-01-01"),
            _raw_game("Xbox One Game", "XOne", "2017-01-01"),
            _raw_game("Series Game", "XSX", "2022-01-01"),
            _raw_game("Switch Game", "NS", "2019-01-01"),
        ]
    )

    enriched = add_derived_features(clean_data(raw)).set_index("title")

    assert str(enriched.loc["PS4 Game", "console_gen"]) == "Eighth Gen"
    assert str(enriched.loc["Xbox One Game", "console_gen"]) == "Eighth Gen"
    assert str(enriched.loc["Switch Game", "console_gen"]) == "Eighth Gen"
    assert str(enriched.loc["PS5 Game", "console_gen"]) == "Ninth Gen"
    assert str(enriched.loc["Series Game", "console_gen"]) == "Ninth Gen"


def test_decade_boundaries_are_calendar_correct():
    raw = pd.DataFrame(
        [
            _raw_game("1989 Game", "NES", "1989-12-31"),
            _raw_game("1990 Game", "SNES", "1990-01-01"),
            _raw_game("2019 Game", "PS4", "2019-12-31"),
            _raw_game("2020 Game", "PS5", "2020-01-01"),
        ]
    )

    enriched = add_derived_features(clean_data(raw)).set_index("title")

    assert str(enriched.loc["1989 Game", "decade"]) == "1980s"
    assert str(enriched.loc["1990 Game", "decade"]) == "1990s"
    assert str(enriched.loc["2019 Game", "decade"]) == "2010s"
    assert str(enriched.loc["2020 Game", "decade"]) == "2020s"


def test_sample_data_uses_real_critic_scale_and_consistent_derived_fields():
    df, _ = create_sample_data()

    scored = df[df["has_critic_score"]]
    unscored = df[~df["has_critic_score"]]

    assert not scored.empty
    assert scored["critic_score"].between(0, 10).all()
    assert np.allclose(
        scored["release_year"].astype(int).to_numpy(),
        scored["release_date"].dt.year.to_numpy(),
    )
    assert np.allclose(
        scored["sales_per_point"].to_numpy(),
        (scored["total_sales"] / scored["critic_score"]).to_numpy(),
        rtol=1e-5,
    )
    assert unscored["critic_score"].isna().all()
    assert unscored["sales_per_point"].isna().all()


def test_narrow_critic_filter_excludes_unrated_games_and_search_is_literal():
    raw = pd.DataFrame(
        [
            _raw_game("Game [A]", "PS4", "2020-01-01", critic_score=9.0),
            _raw_game("Game B", "PS4", "2020-01-01", critic_score=np.nan),
            _raw_game("Game C", "PS4", "2020-01-01", critic_score=7.0),
        ]
    )
    df = add_derived_features(clean_data(raw))
    cache = DataFrameCache()

    filtered = apply_filters(df, cache, critic_range=[8, 10])
    assert list(filtered["title"]) == ["Game [A]"]

    literal = apply_filters(df, cache, search_value="[A]")
    assert list(literal["title"]) == ["Game [A]"]
