# Contributing to Video Game Sales Dashboard

Contributions are welcome. Keep changes focused, testable, and consistent with the analytical definitions documented in `DATA.md`.

## Development setup

1. Fork and clone the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/short-description
   ```
3. Install the development environment:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```
4. Run the dashboard with the bundled dataset:
   ```bash
   python main.py
   ```
   Or use deterministic sample data for a faster smoke test:
   ```bash
   python main.py --sample-data
   ```

## Before opening a pull request

Run the same correctness checks used by GitHub Actions:

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pytest -q
```

You can also inspect the broader, non-blocking style and complexity report with:

```bash
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

For code you change, follow PEP 8, remove unused imports, avoid trailing whitespace, and keep new functions small enough to test independently.

## Data and analytical changes

If a change affects cleaning, derived fields, chart definitions, forecasting, or interpretation:

- update or add tests for the changed behavior;
- update `DATA.md` when field definitions, cleaning rules, or limitations change;
- do not present cumulative lifetime sales as transaction-level monthly or seasonal sales;
- keep missing values explicit rather than replacing them with invented analytical values.

## Pull request process

1. Keep the PR scoped to one coherent change.
2. Update README or data documentation when user-facing behavior changes.
3. Confirm the application initializes on a clean install.
4. Confirm the fatal lint gate and test suite pass.
5. Describe any known limitation or follow-up work in the PR body.

## Reporting bugs

Include the behavior observed, expected behavior, reproduction steps, relevant filters or dataset conditions, and environment details. Screenshots are useful for layout or visualization issues.

## Feature requests

Describe the user problem, expected behavior, and why the proposed feature fits the dashboard's available data.
