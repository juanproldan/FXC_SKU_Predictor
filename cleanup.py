#!/usr/bin/env python
"""
Cleanup script for the SKU prediction project.

This script removes files that are no longer needed after the project reorganization.
"""

import os
import shutil
import sys

# Files to remove
FILES_TO_REMOVE = [
    # Old Python scripts
    "app.py",
    "predict_maker.py",
    "predict_renault_sku.py",
    "predict_sku.py",
    "predict_with_neural.py",
    "predict_with_neural_network.py",
    "predict_with_pytorch.py",
    "analyze_full_dataset.py",
    "analyze_skus.py",
    "check_model.py",
    "check_sku_distribution.py",
    "Extract_sample.py",
    "simple_test.py",
    "simple_train_predict.py",
    "test.py",
    "train_and_predict.py",
    "train_neural_network_model.py",
    "train_pytorch_model.py",
    "train_renault_model.py",
    "train_simple_model.py",
    
    # Temporary files
    "crash_log.txt",
    "pipeline_log.txt",
    "process_full_dataset.log",
    "process_full_dataset_improved.log",
    "database_backup.log",
]

# Directories to remove
DIRS_TO_REMOVE = [
    "src",
    "webapp",
]

def main():
    """Main function to clean up the project."""
    print("Starting project cleanup...")
    
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
