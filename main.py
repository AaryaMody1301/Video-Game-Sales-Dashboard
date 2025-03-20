#!/usr/bin/env python3
"""
Main entry point for the Video Game Sales Dashboard application.

This application provides interactive visualizations and analysis of video game sales
data using Dash and Plotly. It includes features for filtering, forecasting,
and exporting data in various formats.

Usage:
    python main.py [--debug] [--port PORT]

Options:
    --debug     Run the application in debug mode (default: False)
    --port      Specify the port to run the application on (default: 8050)

Examples:
    python main.py
    python main.py --debug --port 8000

Author: Video Game Sales Dashboard Team
License: MIT
"""
import sys
import os
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"dashboard_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Parse command line arguments
def parse_args():
    """Parse command line arguments for the application."""
    parser = argparse.ArgumentParser(description="Video Game Sales Dashboard")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the application on")
    return parser.parse_args()

# Add the project directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app to run it
from src.app import create_app

def main():
    """Main function to start the dashboard application."""
    args = parse_args()
    
    # Create and run the application
    app = create_app()
    
    print(f"Starting dashboard server...")
    print(f"Dash is running on http://127.0.0.1:{args.port}")
    
    # Run the app with the specified options
    app.run_server(debug=args.debug, port=args.port)

if __name__ == '__main__':
    main() 