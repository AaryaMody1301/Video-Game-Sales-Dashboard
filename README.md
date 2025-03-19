# Video Game Sales Dashboard

An interactive dashboard for exploring and visualizing video game sales data using Dash and Plotly.

## Features

- Interactive filtering by year, platform, genre, publisher, and more
- Multiple visualization types including bar charts, pie charts, scatter plots, and heatmaps
- Sales analysis by region, genre, platform, and publisher
- Trend analysis over time
- Game comparison functionality
- Data export capabilities
- Seasonal sales analysis
- Predictive analytics and forecasting
- Responsive design with theme switching

## Project Structure

```
video-game-sales-dashboard/
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contribution guidelines
├── clean.py                    # Project cleanup utility
├── vgchartz-2024.csv           # Dataset (excluded from Git)
└── src/                        # Source code
    ├── app.py                  # Main app initialization
    ├── components/             # UI components
    │   ├── filters.py          # Filter panel
    │   ├── modals.py           # Modal components
    │   └── tabs.py             # Tab components
    ├── data/                   # Data handling
    │   └── data_loader.py      # Data loading and preprocessing
    ├── layouts/                # UI layouts
    │   └── main_layout.py      # Main dashboard layout
    ├── utils/                  # Utilities
    │   └── cache.py            # Caching implementation
    └── callbacks/              # Dashboard interactivity
        ├── comparison_callbacks.py    # Game comparison
        ├── export_callbacks.py        # Data export
        ├── forecast_callbacks.py      # Sales forecasting
        ├── game_details_callbacks.py  # Game details modal
        ├── graph_callbacks.py         # Main visualizations
        ├── register_callbacks.py      # Callback registration
        ├── seasonal_callbacks.py      # Seasonal analysis
        └── theme_callbacks.py         # Theme switching
```

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```
   python main.py
   ```
4. Open your browser and go to `http://localhost:8050`

## Dependencies

- dash
- dash-bootstrap-components
- plotly
- pandas
- numpy
- scikit-learn

## Dataset

The dashboard uses the VGChartz video game sales dataset that includes:
- Game title, platform, genre, publisher, developer
- Release date
- Global and regional sales figures (North America, Japan, Europe, others)
- Critic scores

Note: The dataset file (vgchartz-2024.csv) is not included in the repository due to its size. You'll need to download it separately from [VGChartz](https://www.vgchartz.com/) or use a compatible dataset.

## Development

### Cleaning the Project

To clean up temporary files before committing to GitHub, run:

```
python clean.py
```

This will remove:
- `__pycache__` directories
- `.pyc` files
- Log files

### Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.