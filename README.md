# Video Game Sales Dashboard

A comprehensive data visualization and analysis dashboard for video game sales data.

## Latest Updates

### Code Optimization and Library Updates (2024)

The codebase has been significantly optimized and updated with the following improvements:

- **Updated Dependencies**: All packages updated to their latest versions (as of November 2024)
- **Performance Optimization**: Enhanced data processing, caching, and rendering
- **Async Support**: Added asynchronous operations for better responsiveness
- **Memory Management**: Improved memory usage with adaptive allocation
- **Type Annotations**: Comprehensive typing for better code quality
- **Modern Python Features**: Using latest Python features and best practices

## Features

- Interactive filtering and visualization of video game sales data
- Sales forecasting and trend analysis
- Regional sales comparison
- Publisher and platform performance metrics
- Data export in various formats
- Customizable dashboard with theme selection
- Responsive design for various screen sizes

## Technical Stack

- **Core**: Python 3.11+, Dash 2.17+
- **Data Processing**: Pandas 2.2+, NumPy 1.26+
- **Visualization**: Plotly 5.20+, Dash Bootstrap Components 1.5+
- **Performance**: Numba 0.60+, Cachetools 5.5+
- **Development**: Black, Pytest, MyPy, Flake8

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python main.py
   ```

## Command Line Options

```
python main.py [--debug] [--port PORT] [--host HOST] [--workers WORKERS] [--memory-limit MB] [--cache-size SIZE] [--log-level LEVEL]
```

- `--debug`: Enable debug mode
- `--port`: Port to run the server on (default: 8050)
- `--host`: Host to run the server on (default: 127.0.0.1)
- `--workers`: Number of worker processes (default: 1)
- `--memory-limit`: Memory limit in MB for cache (default: auto-detect)
- `--cache-size`: Maximum items in cache (default: 20)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Performance Enhancements

### Data Processing

- Optimized loading and preprocessing with parallel execution
- JIT compilation for compute-heavy operations with Numba
- Efficient data types with float32 instead of float64
- Automatic data cleaning and standardization

### Memory Management

- Adaptive cache sizing based on system memory
- Intelligent eviction policies using recency and frequency
- Background monitoring for memory pressure
- Automatic garbage collection scheduling

### Asyncio Support

- Asynchronous data loading and processing
- Non-blocking UI updates
- Concurrent request handling
- Compatible with both sync and async environments

## Development

### Project Structure

- `main.py`: Application entry point
- `clean.py`: Data cleanup utilities
- `src/`: Source code directory
  - `app.py`: Application initialization
  - `data/`: Data handling modules
  - `utils/`: Utility functions and classes
  - `components/`: Dashboard UI components
  - `layouts/`: Page layouts
  - `callbacks/`: Interactive functionality

### Contributing

Contributions are welcome! Please follow the coding standards established in the project:

1. Use type hints for all functions
2. Write unit tests for new functionality
3. Follow PEP 8 style guidelines
4. Document all public modules, classes, and functions

## License

MIT

## Acknowledgements

- Data source: VGChartz
- Dashboard design inspired by Plotly Dash Gallery examples