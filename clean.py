#!/python
"""
Clean up temporary files and directories to prepare for Git commits.

This script helps maintain a clean project by removing temporary files,
cache directories, and log files. It supports parallel processing for
large directories and provides a dry-run mode for safety.

Usage:
    python clean.py [--all] [--logs] [--cache] [--dry-run] [--workers N]

Options:
    --all       Remove all temporary files including logs
    --logs      Remove only log files
    --cache     Remove only cache directories
    --dry-run   Show what would be removed without actually removing
    --workers   Number of parallel workers (default: CPU count)
"""

import os
import sys
import shutil
import logging
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

# Windows-specific path handling
def get_project_root() -> Path:
    """Get the project root directory in a cross-platform way."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return Path(sys.executable).parent
    else:
        # Running as script
        return Path(__file__).resolve().parent

def get_log_dir() -> Path:
    """Get the log directory path in a cross-platform way."""
    log_dir = get_project_root() / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir

# Configure logging with proper handler import
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            get_log_dir() / f"cleanup_{datetime.now().strftime('%Y%m%d')}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CleanupConfig:
    """Configuration for cleanup operations."""
    patterns: List[str]
    dry_run: bool
    workers: int
    project_root: Path

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean up project files")
    parser.add_argument("--all", action="store_true", help="Remove all temporary files including logs")
    parser.add_argument("--logs", action="store_true", help="Remove only log files")
    parser.add_argument("--cache", action="store_true", help="Remove only cache directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually removing")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default: CPU count)")
    return parser.parse_args()

def get_cleanup_patterns(args: argparse.Namespace) -> List[str]:
    """Get patterns to clean based on command line arguments."""
    patterns = []
    
    if args.all or args.cache:
        patterns.extend([
            '__pycache__/',
            '.pytest_cache/',
            '.coverage',
            '.hypothesis/',
            '.mypy_cache/',
            '*.pyc',
            '*.pyo',
            '*.pyd',
            '*.so',
            '*.egg-info/',
            'build/',
            'dist/',
            '.eggs/',
            '*.egg'
        ])
    
    if args.all or args.logs:
        patterns.extend([
            '*.log',
            'logs/',
            '*.log.*'
        ])
    
    return patterns

def find_matches(config: CleanupConfig) -> Set[Path]:
    """Find files and directories matching the patterns."""
    matches = set()
    
    def process_path(path: Path) -> List[Path]:
        found = []
        if path.is_dir():
            for pattern in config.patterns:
                if pattern.endswith('/'):
                    if path.name == pattern[:-1]:
                        found.append(path)
                else:
                    found.extend(path.glob(pattern))
        return found
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = []
        for root, dirs, files in os.walk(config.project_root):
            root_path = Path(root)
            futures.append(executor.submit(process_path, root_path))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                matches.update(future.result())
            except Exception as e:
                logger.error(f"Error processing path: {str(e)}")
    
    return matches

def remove_item(item: Path, dry_run: bool) -> None:
    """Remove a file or directory."""
    try:
        if item.is_dir():
            if dry_run:
                logger.info(f"Would remove directory: {item}")
            else:
                shutil.rmtree(item)
                logger.info(f"Removed directory: {item}")
        else:
            if dry_run:
                logger.info(f"Would remove file: {item}")
            else:
                item.unlink()
                logger.info(f"Removed file: {item}")
    except Exception as e:
        logger.error(f"Error removing {item}: {str(e)}")

def remove_items(items: Set[Path], config: CleanupConfig) -> None:
    """Remove items in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = [
            executor.submit(remove_item, item, config.dry_run)
            for item in items
        ]
        concurrent.futures.wait(futures)

def main() -> None:
    """Main function to clean up the project."""
    try:
        # Parse arguments
        args = parse_args()
        if not any([args.all, args.logs, args.cache]):
            logger.info("No cleanup options specified. Use --all, --logs, or --cache to specify what to clean.")
            return
        
        # Setup configuration
        config = CleanupConfig(
            patterns=get_cleanup_patterns(args),
            dry_run=args.dry_run,
            workers=args.workers or os.cpu_count() or 1,
            project_root=get_project_root()
        )
        
        # Find matches
        logger.info("Scanning for files to remove...")
        matches = find_matches(config)
        
        if not matches:
            logger.info("No files found matching the specified patterns.")
            return
        
        # Print summary
        logger.info(f"Found {len(matches)} items to remove:")
        for item in matches:
            logger.info(f"  {item}")
        
        # Confirm and remove
        if config.dry_run:
            logger.info("Dry run: no files were actually removed")
        else:
            if input("\nProceed with removal? (y/n): ").lower() == 'y':
                remove_items(matches, config)
                logger.info("Cleanup completed successfully!")
            else:
                logger.info("Operation cancelled.")
    
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 