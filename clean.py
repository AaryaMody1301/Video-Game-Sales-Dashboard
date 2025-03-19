import os
import shutil
import glob

def clean_project():
    """Clean up unnecessary files and directories from the project."""
    print("Cleaning project...")
    
    # Remove __pycache__ directories
    for pycache_dir in glob.glob("**/__pycache__", recursive=True):
        if os.path.isdir(pycache_dir):
            print(f"Removing: {pycache_dir}")
            shutil.rmtree(pycache_dir)
    
    # Remove .pyc files
    for pyc_file in glob.glob("**/*.pyc", recursive=True):
        if os.path.isfile(pyc_file):
            print(f"Removing: {pyc_file}")
            os.remove(pyc_file)
    
    # Remove log files
    for log_file in glob.glob("**/*.log", recursive=True):
        if os.path.isfile(log_file):
            print(f"Removing: {log_file}")
            os.remove(log_file)
    
    print("Project cleaned successfully!")

if __name__ == "__main__":
    clean_project() 