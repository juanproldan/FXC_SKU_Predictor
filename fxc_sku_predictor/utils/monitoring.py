"""
Log monitoring utilities for the SKU prediction system.

This module provides functions for monitoring log files for errors
and sending notifications when errors are detected.
"""

import os
import sys
import re
import time
import logging
import argparse
import smtplib
import json
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Log directory
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Email configuration (update with your email settings)
EMAIL_CONFIG = {
    'enabled': False,  # Set to True to enable email notifications
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'your_email@gmail.com',
    'password': 'your_app_password',  # Use app password for Gmail
    'from_email': 'your_email@gmail.com',
    'to_email': 'recipient@example.com',
    'subject_prefix': '[SKU Predictor Monitor]'
}

# Log files to monitor
LOG_FILES = {
    'app': os.path.join(LOG_DIR, 'app.log'),
    'retraining': os.path.join(LOG_DIR, 'retraining_scheduler.log'),
    'database': os.path.join(LOG_DIR, 'database_backup.log')
}

# Error patterns to look for
ERROR_PATTERNS = [
    r'ERROR',
    r'Exception',
    r'Error:',
    r'Failed',
    r'Traceback',
    r'Critical'
]

def send_email_notification(subject, message):
    """Send an email notification.
    
    Args:
        subject (str): The subject of the email.
        message (str): The body of the email.
        
    Returns:
        bool: True if the email was sent successfully, False otherwise.
    """
    if not EMAIL_CONFIG['enabled']:
        logger.info("Email notifications are disabled.")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['from_email']
        msg['To'] = EMAIL_CONFIG['to_email']
        msg['Subject'] = f"{EMAIL_CONFIG['subject_prefix']} {subject}"
        
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['username'], EMAIL_CONFIG['password'])
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email notification sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email notification: {str(e)}")
        return False

def setup_app_logging():
    """Set up logging for the Flask application.
    
    Returns:
        logging.Logger: The configured logger.
    """
    # Create a symbolic link from app.py's log to our monitored log directory
    app_log_path = os.path.join(LOG_DIR, 'app.log')
    
    # Add a handler to the Flask app's logger
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)
    
    # Create a file handler
    handler = logging.FileHandler(app_log_path)
    handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    app_logger.addHandler(handler)
    
    logger.info(f"Set up application logging to {app_log_path}")
    return app_logger

def check_log_file(log_file, last_position=0, hours=24):
    """Check a log file for errors since the last check.
    
    Args:
        log_file (str): Path to the log file.
        last_position (int): Position in the file to start reading from.
        hours (int): Number of hours of logs to check.
        
    Returns:
        tuple: (new_position, errors) where new_position is the position to start
            reading from next time, and errors is a list of error lines.
    """
    if not os.path.exists(log_file):
        logger.warning(f"Log file not found: {log_file}")
        return last_position, []
    
    # Get file size
    file_size = os.path.getsize(log_file)
    
    # If file was truncated or rotated, start from beginning
    if file_size < last_position:
        last_position = 0
    
    # If this is the first check, only check the last part of the file
    if last_position == 0 and file_size > 10000:
        last_position = max(0, file_size - 10000)  # Start from the last 10KB
    
    # Calculate the cutoff time
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    errors = []
    with open(log_file, 'r') as f:
        f.seek(last_position)
        for line in f:
            # Try to extract timestamp from the line
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                try:
                    line_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                    if line_time < cutoff_time:
                        continue  # Skip old entries
                except ValueError:
                    pass  # If timestamp parsing fails, check the line anyway
            
            # Check if line contains any error patterns
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in ERROR_PATTERNS):
                errors.append(line.strip())
        
        # Update the last position
        last_position = f.tell()
    
    return last_position, errors

def monitor_logs(args):
    """Monitor log files for errors.
    
    Args:
        args: Command line arguments.
        
    Returns:
        dict: Dictionary mapping log names to lists of error lines.
    """
    # Load last positions from file if it exists
    last_positions = {}
    positions_file = os.path.join(LOG_DIR, 'last_positions.json')
    
    if os.path.exists(positions_file):
        try:
            with open(positions_file, 'r') as f:
                last_positions = json.load(f)
        except Exception as e:
            logger.error(f"Error loading last positions: {str(e)}")
    
    # Check each log file
    all_errors = {}
    for log_name, log_file in LOG_FILES.items():
        last_pos = last_positions.get(log_name, 0)
        new_pos, errors = check_log_file(log_file, last_pos, args.hours)
        
        # Update last position
        last_positions[log_name] = new_pos
        
        if errors:
            all_errors[log_name] = errors
            logger.info(f"Found {len(errors)} errors in {log_name} log")
    
    # Save last positions
    try:
        with open(positions_file, 'w') as f:
            json.dump(last_positions, f)
    except Exception as e:
        logger.error(f"Error saving last positions: {str(e)}")
    
    # Send notification if errors were found
    if all_errors and args.notify:
        message = "The following errors were detected in the logs:\n\n"
        
        for log_name, errors in all_errors.items():
            message += f"=== {log_name.upper()} LOG ===\n"
            message += "\n".join(errors[:10])  # Limit to first 10 errors
            if len(errors) > 10:
                message += f"\n... and {len(errors) - 10} more errors\n"
            message += "\n\n"
        
        send_email_notification("Errors Detected in Logs", message)
    
    return all_errors

def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Monitor log files for errors.')
    parser.add_argument('--hours', type=int, default=24, help='Hours of logs to check')
    parser.add_argument('--notify', action='store_true', help='Send email notifications')
    parser.add_argument('--setup', action='store_true', help='Set up application logging')
    parser.add_argument('--continuous', action='store_true', help='Run in continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=3600, help='Check interval in seconds (for continuous mode)')
    return parser.parse_args()

def main():
    """Main function."""
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, 'monitor.log')),
            logging.StreamHandler()
        ]
    )
    
    args = parse_arguments()
    
    if args.setup:
        setup_app_logging()
        return
    
    logger.info(f"Starting log monitor (checking last {args.hours} hours)")
    
    if args.continuous:
        logger.info(f"Running in continuous mode, checking every {args.interval} seconds")
        try:
            while True:
                errors = monitor_logs(args)
                if errors:
                    logger.warning(f"Found errors in logs: {sum(len(e) for e in errors.values())} total errors")
                else:
                    logger.info("No errors found in logs")
                
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")
    else:
        errors = monitor_logs(args)
        if errors:
            logger.warning(f"Found errors in logs: {sum(len(e) for e in errors.values())} total errors")
            sys.exit(1)  # Exit with error code
        else:
            logger.info("No errors found in logs")

if __name__ == "__main__":
    main()
