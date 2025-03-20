#!/usr/bin/env python3
"""
Clean up temporary files and directories to prepare for Git commits.
"""

import os
import shutil
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Clean up project files")
    parser.add_argument("--all", action="store_true", help="Remove all temporary files including logs")
    parser.add_argument("--logs", action="store_true", help="Remove only log files")
    parser.add_argument("--cache", action="store_true", help="Remove only cache directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually removing")
    return parser.parse_args()

def find_pattern(directory, patterns):
    """Find files/directories matching the given patterns"""
    all_matches = []
    for pattern in patterns:
        if pattern.endswith('/'):  # Directory pattern
            for root, dirs, _ in os.walk(directory):
                for d in dirs:
                    if d == pattern[:-1]:
                        all_matches.append(os.path.join(root, d))
        else:  # File pattern
            for root, _, files in os.walk(directory):
                for f in files:
                    if f.endswith(pattern):
                        all_matches.append(os.path.join(root, f))
    return all_matches

def remove_items(items, dry_run=False):
    """Remove the specified items"""
    for item in items:
        if os.path.isdir(item):
            logger.info(f"Removing directory: {item}")
            if not dry_run:
                shutil.rmtree(item)
        else:
            logger.info(f"Removing file: {item}")
            if not dry_run:
                os.remove(item)

def main():
    """Main function to clean up the project"""
    args = parse_args()
    project_root = Path(__file__).parent
    
    # Define patterns to remove
    cache_patterns = ['__pycache__/', '.pytest_cache/']
    python_cache_patterns = ['.pyc', '.pyo', '.pyd']
    log_patterns = ['.log']
    
    to_remove = []
    
    if args.all or args.cache:
        # Find and remove cache directories
        cache_matches = find_pattern(project_root, cache_patterns)
        to_remove.extend(cache_matches)
        
        # Find and remove Python cache files
        python_cache_matches = find_pattern(project_root, python_cache_patterns)
        to_remove.extend(python_cache_matches)
    
    if args.all or args.logs:
        # Find and remove log files
        log_matches = find_pattern(project_root, log_patterns)
        to_remove.extend(log_matches)
    
    if not to_remove:
        logger.info("No files to remove. Use --all, --logs, or --cache to specify what to clean.")
        return
    
    # Print summary
    logger.info(f"Found {len(to_remove)} items to remove:")
    for item in to_remove:
        logger.info(f"  {item}")
    
    # Confirm and remove
    if args.dry_run:
        logger.info("Dry run: no files were actually removed")
    else:
        if input("\nProceed with removal? (y/n): ").lower() == 'y':
            remove_items(to_remove)
            logger.info("Cleanup completed successfully!")
        else:
            logger.info("Operation cancelled.")

if __name__ == "__main__":
    main() 