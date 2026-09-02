import pandas as pd

from src.callbacks.graph_callbacks import (
    _build_figures,
    _publisher_share_figure,
    _release_year_figure,
)


def _sample_frame():
    return pd.DataFrame(
        {
            "title": ["A", "B", "C"],
            "console": ["PS5", "PS5", "PC"],
            "genre": ["Action", "RPG", "Action"],
            "publisher": ["Pub A", "Pub B", "Pub C"],
            "release_year": [2020, 2021, 2021],
            "console_gen": ["Ninth Gen", "Ninth Gen", "PC"],
            "total_sales": [10.0, 5.0, 2.0],
            "na_sales": [4.0, 2.0, 1.0],
            "jp_sales": [1.0, 1.0, 0.2],
            "pal_sales": [4.0, 1.5, 0.6],
            "other_sales": [1.0, 0.5, 0.2],
            "critic_score": [8.0, 7.0, 9.0],
            "sales_per_point": [1.25, 5.0 / 7.0, 2.0 / 9.0],
        }
    )


def test_publisher_share_includes_other_and_preserves_total_sales():
    frame = _sample_frame()

    figure = _publisher_share_figure(frame, display_count=2)

    labels = list(figure.data[0]["labels"])
    values = list(figure.data[0]["values"])
    assert "Other" in labels
    assert sum(values) == frame["total_sales"].sum()


def test_release_year_title_describes_release_cohorts():
    figure = _release_year_figure(_sample_frame(), simple_charts=False)

    assert figure.layout.title.text == "Lifetime Sales by Game Release Year"


def test_graph_builder_returns_all_expected_figures():
    figures = _build_figures(
        _sample_frame(),
        sort_method="total_sales",
        display_count=2,
        simple_charts=False,
        use_custom_templates=True,
        missing_platforms=[],
    )

    assert len(figures) == 12
