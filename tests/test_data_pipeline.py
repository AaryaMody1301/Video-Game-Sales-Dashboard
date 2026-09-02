import pandas as pd

from src.data.data_loader import apply_filters, clean_data, create_sample_data


def test_create_sample_data_has_expected_shape_and_columns():
    df, cache = create_sample_data()

    required_columns = {
        "title",
        "console",
        "publisher",
        "developer",
        "genre",
        "release_date",
        "total_sales",
        "critic_score",
        "release_year",
        "console_gen",
        "release_month",
        "release_quarter",
    }

    assert len(df) == 30
    assert required_columns.issubset(df.columns)
    assert cache is not None


def test_clean_data_removes_duplicates_and_zero_sales_rows():
    raw = pd.DataFrame(
        [
            {
                "title": "Game A",
                "console": "PC",
                "publisher": "Publisher",
                "developer": "Developer",
                "genre": "Action",
                "release_date": "2020-01-01",
                "total_sales": 2.0,
                "na_sales": 1.0,
                "jp_sales": 0.2,
                "pal_sales": 0.6,
                "other_sales": 0.2,
                "critic_score": 8.0,
            },
            {
                "title": "Game A",
                "console": "PC",
                "publisher": "Publisher",
                "developer": "Developer",
                "genre": "Action",
                "release_date": "2020-01-01",
                "total_sales": 2.0,
                "na_sales": 1.0,
                "jp_sales": 0.2,
                "pal_sales": 0.6,
                "other_sales": 0.2,
                "critic_score": 8.0,
            },
            {
                "title": "Game B",
                "console": "PC",
                "publisher": "Publisher",
                "developer": "Developer",
                "genre": "Action",
                "release_date": "2021-01-01",
                "total_sales": 0.0,
                "na_sales": 0.0,
                "jp_sales": 0.0,
                "pal_sales": 0.0,
                "other_sales": 0.0,
                "critic_score": None,
            },
        ]
    )

    cleaned = clean_data(raw)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["title"] == "Game A"


def test_apply_filters_uses_search_and_platform_filters():
    df, cache = create_sample_data()
    target = df.iloc[0]

    filtered = apply_filters(
        df,
        cache,
        selected_platforms=[str(target["console"])],
        search_value=str(target["title"]),
    )

    assert not filtered.empty
    assert set(filtered["console"].astype(str)) == {str(target["console"])}
    assert all(str(target["title"]).lower() in title.lower() for title in filtered["title"])
