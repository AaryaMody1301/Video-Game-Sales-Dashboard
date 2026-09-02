# Video Game Sales Dashboard

[![Python Application](https://github.com/AaryaMody1301/Video-Game-Sales-Dashboard/actions/workflows/python-app.yml/badge.svg)](https://github.com/AaryaMody1301/Video-Game-Sales-Dashboard/actions/workflows/python-app.yml)

An interactive Dash application for exploring reported video game sales across platforms, publishers, genres, regions, release periods, and critic scores. The project combines a cleaned 64K-game source dataset with interactive visual analysis, comparison tools, export workflows, and exploratory forecasting.

## Highlights

- Interactive filtering by release year, platform, console generation, genre, publisher, critic score, and title.
- Platform, genre, regional, publisher, critic-score, and release-cohort visualizations.
- Linear, polynomial, ridge, and ARIMA forecasting with chronological validation.
- Stable game identities for comparing same-title releases across platforms.
- CSV and Excel export of the currently filtered dataset.
- Release-season analysis that clearly distinguishes release timing from transaction timing.
- Responsive Bootstrap layout and switchable visual themes.
- Deterministic sample-data mode for quick evaluation without relying on the full CSV.
- Automated CI with dependency installation, fatal lint checks, and pytest regression coverage.

## Tech Stack

- Python 3.11
- Dash 2.17 + Dash Bootstrap Components
- pandas + NumPy
- Plotly
- scikit-learn
- statsmodels
- Numba
- pytest + GitHub Actions

## Quick Start

```bash
git clone https://github.com/AaryaMody1301/Video-Game-Sales-Dashboard.git
cd Video-Game-Sales-Dashboard
python -m venv .venv
```

Activate the virtual environment, then install the runtime dependencies:

```bash
pip install -r requirements.txt
```

Run with the repository dataset:

```bash
python main.py
```

Or run immediately with deterministic built-in sample data:

```bash
python main.py --sample-data
```

Open `http://127.0.0.1:8050` in a browser.

## Command-Line Options

```text
python main.py [--debug] [--port PORT] [--host HOST] [--workers WORKERS]
               [--memory-limit MB] [--cache-size SIZE] [--log-level LEVEL]
               [--disable-custom-templates] [--simple-charts] [--sample-data]
```

Examples:

```bash
python main.py --debug
python main.py --host 0.0.0.0 --port 8050
python main.py --sample-data --cache-size 10
```

## Project Structure

```text
.
├── .github/workflows/python-app.yml
├── DATA.md
├── main.py
├── requirements.txt
├── requirements-dev.txt
├── src/
│   ├── app.py
│   ├── callbacks/
│   ├── components/
│   ├── data/
│   ├── layouts/
│   └── utils/
├── tests/
└── vgchartz-2024.csv
```

### Architecture

`main.py` owns CLI/server configuration. `src/app.py` contains one synchronous Dash application factory. The data layer loads and cleans the source dataset, then callbacks use a bounded thread-safe LRU cache for repeated filter results. UI components and callbacks remain separated so analytical logic can be regression-tested independently of the page layout.

The application intentionally avoids background cache-monitor or performance-monitor threads. Cache bounds are enforced during normal reads/writes, which keeps startup, tests, and process shutdown deterministic.

## Data and Interpretation

The project uses the **Video Game Sales 2024** dataset published on Kaggle and sourced from VGChartz collection work. Kaggle lists the dataset under the ODC Attribution License (ODC-By).

See [DATA.md](DATA.md) for the data dictionary, cleaning rules, provenance, and analytical limitations.

A key limitation is that the dataset contains **reported cumulative lifetime sales plus release dates**, not transaction-level monthly/yearly sales. Consequently, time-based charts show lifetime sales grouped by the release period of each game. The release-season tab describes performance of games released in a given month or quarter; it does not claim that those sales occurred in that period.

## Forecasting

The Predictive Analytics tab supports:

- Linear regression
- Polynomial regression
- Ridge regression
- ARIMA

Validation is chronological rather than randomly shuffled. Regression models use time-aware cross-validation, ARIMA uses the current statsmodels API, forecasts are constrained to non-negative sales, and a selected model is never silently replaced with a different model.

These forecasts are exploratory extrapolations of release-cohort sales totals and should not be interpreted as financial forecasts.

## Development

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```

Run the fatal CI lint gate locally:

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

The workflow also reports broader style/complexity warnings without blocking CI so correctness failures remain distinct from cleanup debt.

## Reliability Work Completed

The repository has been hardened in four stages:

1. **CI and test foundation** — activated CI for `master`, fixed dependency installation, and added regression coverage.
2. **Analytical correctness** — corrected date semantics, decade/generation mapping, score handling, filter bounds, and sample-data consistency.
3. **Advanced-feature reliability** — repaired ARIMA and time-aware validation, game identity handling, modal selection, and export behavior.
4. **Portfolio/runtime cleanup** — simplified app/cache lifecycle, removed pseudo-async startup, added responsive layout, separated runtime/dev dependencies, and documented dataset limitations.

## License

Project code is licensed under the MIT License. Dataset licensing and attribution are documented separately in [DATA.md](DATA.md).
