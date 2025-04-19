#!/usr/bin/env python
"""
Cleanup script for the Fuente_Json_Consolidado directory.

This script removes files that are not needed for the current system.
"""

import os
import shutil
import sys

# Files to keep
FILES_TO_KEEP = [
    "Consolidado.json",           # Main data source
    "clean_training_data.csv",    # Generated from Consolidado.json and used for training
    "item_training_data.csv",     # Used by the training scripts
    "maker_series_model.json",    # Used for the web application's dropdown menus
]

# Directory to clean
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "Fuente_Json_Consolidado")

def main():
    """Main function to clean up the data directory."""
    print(f"Starting cleanup of {DATA_DIR}...")
    
    # Get all files in the directory
    all_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    
    # Identify files to remove
    files_to_remove = [f for f in all_files if f not in FILES_TO_KEEP]
    
    # Get confirmation from the user
    print("\nThe following files will be removed:")
    for file in files_to_remove:
        print(f"  - {file}")
    
    print("\nThe following files will be kept:")
    for file in FILES_TO_KEEP:
        if file in all_files:
            print(f"  - {file}")
    
    confirmation = input("\nAre you sure you want to remove these files? (y/n): ")
    if confirmation.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    # Remove files
    removed_files = 0
    for file in files_to_remove:
        file_path = os.path.join(DATA_DIR, file)
        try:
            os.remove(file_path)
            print(f"Removed file: {file}")
            removed_files += 1
        except Exception as e:
            print(f"Error removing file {file}: {str(e)}")
    
    print(f"\nCleanup complete. Removed {removed_files} files.")

if __name__ == "__main__":
    main()
