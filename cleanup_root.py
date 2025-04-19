#!/usr/bin/env python
"""
Cleanup script for the root directory.

This script removes files and directories that are not needed for the current system.
"""

import os
import shutil
import sys

# Files to remove
FILES_TO_REMOVE = [
    "sample.json",               # Sample file used during development
    "test_single_input.csv",     # Test file
    "test_feedback.py",          # Functionality moved to the package
    "TODO_SKU_desde_descripcion.md",  # Outdated now that the project is complete
]

# Directories to remove
DIRS_TO_REMOVE = [
    "src",                       # Contains only __pycache__
    "notebooks",                 # Empty directory
    "tests",                     # Contains only an empty __init__.py file
]

def main():
    """Main function to clean up the root directory."""
    print("Starting cleanup of root directory...")
    
    # Get confirmation from the user
    print("\nThe following files will be removed:")
    for file in FILES_TO_REMOVE:
        if os.path.exists(file):
            print(f"  - {file}")
    
    print("\nThe following directories will be removed:")
    for directory in DIRS_TO_REMOVE:
        if os.path.exists(directory):
            print(f"  - {directory}")
    
    confirmation = input("\nAre you sure you want to remove these files and directories? (y/n): ")
    if confirmation.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    # Remove files
    removed_files = 0
    for file in FILES_TO_REMOVE:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Removed file: {file}")
                removed_files += 1
            except Exception as e:
                print(f"Error removing file {file}: {str(e)}")
    
    # Remove directories
    removed_dirs = 0
    for directory in DIRS_TO_REMOVE:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Removed directory: {directory}")
                removed_dirs += 1
            except Exception as e:
                print(f"Error removing directory {directory}: {str(e)}")
    
    print(f"\nCleanup complete. Removed {removed_files} files and {removed_dirs} directories.")

if __name__ == "__main__":
    main()
