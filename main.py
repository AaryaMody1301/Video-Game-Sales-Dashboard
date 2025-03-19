"""
Main entry point for the Video Game Sales Dashboard application

This application provides visualizations and analysis of video game sales
data using Dash and Plotly.
"""
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app to run it
from src.app import main

if __name__ == '__main__':
    main() 