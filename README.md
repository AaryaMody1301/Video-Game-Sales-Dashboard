# Video Game Sales Dashboard

An interactive data visualization and forecasting dashboard for analyzing video game sales trends.

## Features

- **Interactive Data Exploration**: Filter and visualize video game sales data by year, platform, genre, publisher, and more
- **Advanced Sales Forecasting**: Predict future sales trends using multiple models:
  - Linear Regression
  - Polynomial Regression
  - Ridge Regression
  - ARIMA Time Series Analysis
- **Genre Analysis**: Visualize sales distribution across different game genres
- **Seasonal Trends**: Analyze how sales vary throughout the year
- **Platform Comparison**: Compare sales performance across different gaming platforms
- **Publisher Insights**: Discover top publishers and their market share
- **Data Export**: Download visualizations and filtered data

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/video-game-sales-dashboard.git
cd video-game-sales-dashboard
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python main.py
```

4. Open your browser and navigate to `http://127.0.0.1:8050`

## Usage

1. **Data Filtering**: Use the filter panel to select specific platforms, genres, years, or publishers
2. **Visualization**: Explore different tabs to view various charts and insights
3. **Forecasting**: Navigate to the Forecast tab, set parameters, and generate sales projections
4. **Export**: Download visualizations or filtered data using the export buttons

## Technical Details

### Architecture

The dashboard is built with:
- **Dash**: Web application framework for interactive data visualization
- **Plotly**: Interactive plotting library
- **Pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning models for forecasting
- **statsmodels**: Time series analysis and ARIMA modeling

### File Structure

```
├── main.py                   # Application entry point
├── requirements.txt          # Project dependencies
├── src/
│   ├── app.py                # Dash application configuration
│   ├── components/           # UI components
│   ├── callbacks/            # Interactive callback functions
│   ├── data/                 # Data loading and processing
│   └── utils/                # Utility functions
├── assets/                   # Static assets (CSS, images)
└── data/                     # Dataset files
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.