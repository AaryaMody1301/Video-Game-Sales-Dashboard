# Contributing to Video Game Sales Dashboard

Thank you for considering contributing to the Video Game Sales Dashboard project! This document outlines the process for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
   ```
   git clone https://github.com/YOUR-USERNAME/Video-Game-Sales-Dashboard.git
   ```
3. Create a virtual environment and install dependencies
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
4. Create a branch for your changes
   ```
   git checkout -b feature/your-feature-name
   ```

## Development Process

1. Make your changes
2. Run the cleaning script to remove temporary files
   ```
   python clean.py
   ```
3. Test your changes locally
   ```
   python main.py
   ```
4. Commit your changes with a descriptive commit message
   ```
   git commit -m "Add feature: description of your changes"
   ```
5. Push your changes to your fork
   ```
   git push origin feature/your-feature-name
   ```
6. Create a Pull Request on GitHub

## Code Style

- Follow PEP 8 for Python code style
- Use docstrings for functions and classes
- Keep functions small and focused on a single task
- Write descriptive variable and function names

## Adding Features

When adding new features:

1. Make sure they fit within the scope of the project
2. Update documentation to reflect the new feature
3. If adding new dependencies, update requirements.txt
4. Maintain the existing code organization structure

## Reporting Issues

If you encounter a bug or have a feature request:

1. Check if the issue already exists in the GitHub Issues
2. If not, create a new issue with a descriptive title and detailed description
3. Include steps to reproduce the issue if applicable
4. Mention your operating system and Python version

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license. 