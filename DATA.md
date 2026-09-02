# Dataset Notes

## Source

This project uses the **Video Game Sales 2024** dataset distributed on Kaggle:

- Dataset: https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024
- Reported source: data collected from VGChartz using the collection methodology described on the Kaggle dataset page.
- Dataset license listed by Kaggle: **ODC Attribution License (ODC-By)**.

The repository currently keeps `vgchartz-2024.csv` as the reproducible project snapshot used by the dashboard.

## Fields used by the dashboard

| Field | Meaning |
| --- | --- |
| `title` | Game title |
| `console` | Platform / console |
| `genre` | Game genre |
| `publisher` | Publisher |
| `developer` | Developer |
| `critic_score` | Critic score on a 0-10 scale when available |
| `total_sales` | Reported lifetime global sales in millions |
| `na_sales` | Reported lifetime North American sales in millions |
| `jp_sales` | Reported lifetime Japanese sales in millions |
| `pal_sales` | Reported lifetime PAL-region sales in millions |
| `other_sales` | Reported lifetime sales in other regions in millions |
| `release_date` | Game release date when available |

## Cleaning rules

The loader:

- preserves unknown release dates as missing values rather than inventing a date;
- preserves missing critic scores as missing values;
- removes duplicate title/console/release-date rows;
- uses the regional sum when total sales is absent but regional values exist;
- prevents total sales from being lower than the sum of reported regional sales;
- removes rows where all sales fields are zero;
- standardizes selected publisher names; and
- derives release year/month/quarter/decade, console generation, regional shares, and score-based metrics.

## Analytical limitations

The dataset contains cumulative reported game sales associated with a game's release date. It does **not** contain transaction-level monthly or yearly sales observations.

Therefore:

- charts grouped by release year show **lifetime sales of games released in that year**, not sales transactions that occurred during that year;
- release-month and release-quarter charts describe **release-season performance**, not the month or quarter in which purchases occurred;
- forecasts extrapolate historical release-cohort totals and should be treated as exploratory, not as financial or market forecasts;
- missing critic scores are excluded from score-specific calculations rather than imputed; and
- reported sales coverage varies by title and platform, so totals should be interpreted as dataset-reported values rather than audited market totals.
