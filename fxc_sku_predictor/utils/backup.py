"""
Database backup utilities for the SKU prediction system.

This module provides functions for creating, listing, and restoring
backups of the feedback database.
"""

import os
import sys
import shutil
import sqlite3
import logging
import datetime
import zipfile
import argparse
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Import the database path
from fxc_sku_predictor.core.feedback_db import DB_PATH

# Backup directory
BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'backups')
os.makedirs(BACKUP_DIR, exist_ok=True)

def create_backup():
    """Create a backup of the feedback database.
    
    Returns:
        bool: True if the backup was successful, False otherwise.
    """
    try:
        # Check if the database exists
        if not os.path.exists(DB_PATH):
            logger.error(f"Database file not found at {DB_PATH}")
            return False
        
        # Create a timestamp for the backup filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"feedback_db_backup_{timestamp}.db"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        
        # Connect to the database to ensure it's not locked
        conn = sqlite3.connect(DB_PATH)
        conn.close()
        
        # Copy the database file
        shutil.copy2(DB_PATH, backup_path)
        logger.info(f"Database backup created at {backup_path}")
        
        # Create a zip archive of the backup
        zip_path = f"{backup_path}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(backup_path, os.path.basename(backup_path))
        logger.info(f"Backup compressed to {zip_path}")
        
        # Remove the uncompressed backup
        os.remove(backup_path)
        
        # Clean up old backups (keep only the 10 most recent)
        cleanup_old_backups()
        
        return True
    except Exception as e:
        logger.error(f"Error creating database backup: {str(e)}")
        return False

def cleanup_old_backups(keep_count=10):
    """Remove old backups, keeping only the most recent ones.
    
    Args:
        keep_count (int): Number of backups to keep.
        
    Returns:
        int: Number of backups removed.
    """
    try:
        # Get all zip files in the backup directory
        backup_files = [f for f in os.listdir(BACKUP_DIR) if f.endswith('.zip') and f.startswith('feedback_db_backup_')]
        
        # Sort by filename (which includes timestamp)
        backup_files.sort(reverse=True)
        
        # Remove old backups
        removed_count = 0
        for old_file in backup_files[keep_count:]:
            os.remove(os.path.join(BACKUP_DIR, old_file))
            logger.info(f"Removed old backup: {old_file}")
            removed_count += 1
            
        return removed_count
    except Exception as e:
        logger.error(f"Error cleaning up old backups: {str(e)}")
        return 0

def restore_backup(backup_path):
    """Restore the database from a backup.
    
    Args:
        backup_path (str): Path to the backup file.
        
    Returns:
        bool: True if the restore was successful, False otherwise.
    """
    try:
        # Check if the backup exists
        if not os.path.exists(backup_path):
            logger.error(f"Backup file not found at {backup_path}")
            return False
        
        # If it's a zip file, extract it first
        if backup_path.endswith('.zip'):
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # Extract to a temporary location
                temp_dir = os.path.join(BACKUP_DIR, 'temp')
                os.makedirs(temp_dir, exist_ok=True)
                zipf.extractall(temp_dir)
                
                # Find the extracted database file
                extracted_files = [f for f in os.listdir(temp_dir) if f.endswith('.db')]
                if not extracted_files:
                    logger.error(f"No database file found in the backup zip")
                    return False
                
                # Use the first database file found
                extracted_path = os.path.join(temp_dir, extracted_files[0])
                
                # Copy to the original location
                shutil.copy2(extracted_path, DB_PATH)
                
                # Clean up
                shutil.rmtree(temp_dir)
        else:
            # Direct copy if it's not a zip file
            shutil.copy2(backup_path, DB_PATH)
        
        logger.info(f"Database restored from {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Error restoring database: {str(e)}")
        return False

def list_backups():
    """List all available backups.
    
    Returns:
        list: List of backup filenames.
    """
    try:
        backup_files = [f for f in os.listdir(BACKUP_DIR) if f.endswith('.zip') and f.startswith('feedback_db_backup_')]
        backup_files.sort(reverse=True)
        
        if not backup_files:
            logger.info("No backups found")
            return []
        
        logger.info(f"Found {len(backup_files)} backups:")
        for i, backup in enumerate(backup_files):
            # Extract timestamp from filename
            timestamp_str = backup.replace('feedback_db_backup_', '').replace('.db.zip', '')
            try:
                timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = "Unknown date"
            
            # Get file size
            file_path = os.path.join(BACKUP_DIR, backup)
            size_bytes = os.path.getsize(file_path)
            size_mb = size_bytes / (1024 * 1024)
            
            logger.info(f"{i+1}. {backup} - {formatted_time} - {size_mb:.2f} MB")
        
        return backup_files
    except Exception as e:
        logger.error(f"Error listing backups: {str(e)}")
        return []

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Database backup utility')
    parser.add_argument('--create', action='store_true', help='Create a new backup')
    parser.add_argument('--list', action='store_true', help='List available backups')
    parser.add_argument('--restore', type=str, help='Restore from a backup file')
    parser.add_argument('--cleanup', type=int, help='Clean up old backups, keeping N most recent')
    return parser.parse_args()

def main():
    """Main function to run the backup process."""
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                            'logs', 'database_backup.log')),
            logging.StreamHandler()
        ]
    )
    
    args = parse_args()
    
    if args.create:
        logger.info("Starting database backup process...")
        if create_backup():
            logger.info("Database backup completed successfully")
        else:
            logger.error("Database backup failed")
    
    if args.list or (not args.create and not args.restore and not args.cleanup):
        logger.info("Available backups:")
        list_backups()
    
    if args.restore:
        logger.info(f"Restoring from backup: {args.restore}")
        if restore_backup(args.restore):
            logger.info("Database restore completed successfully")
        else:
            logger.error("Database restore failed")
    
    if args.cleanup:
        logger.info(f"Cleaning up old backups, keeping {args.cleanup} most recent...")
        removed = cleanup_old_backups(args.cleanup)
        logger.info(f"Removed {removed} old backups")

if __name__ == "__main__":
    main()
