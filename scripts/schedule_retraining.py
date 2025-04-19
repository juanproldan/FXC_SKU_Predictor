from backup_database import create_backup
import os
import sys
import time
import subprocess
import logging
import smtplib
import traceback
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import project modules
try:
    import schedule
except ImportError:
    logging.error(
        "Schedule module not found. Please install it with 'pip install schedule'")
    sys.exit(1)

# Import our custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "retraining_scheduler.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Path to the retraining script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RETRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "retrain_with_feedback.py")

# Email configuration (update with your email settings)
EMAIL_CONFIG = {
    'enabled': False,  # Set to True to enable email notifications
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'your_email@gmail.com',
    'password': 'your_app_password',  # Use app password for Gmail
    'from_email': 'your_email@gmail.com',
    'to_email': 'recipient@example.com',
    'subject_prefix': '[SKU Predictor]'
}


def send_email_notification(subject, message):
    """Send an email notification."""
    if not EMAIL_CONFIG['enabled']:
        logger.info("Email notifications are disabled.")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['from_email']
        msg['To'] = EMAIL_CONFIG['to_email']
        msg['Subject'] = f"{EMAIL_CONFIG['subject_prefix']} {subject}"

        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(
            EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['username'], EMAIL_CONFIG['password'])
        server.send_message(msg)
        server.quit()

        logger.info(f"Email notification sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email notification: {str(e)}")
        return False


def backup_database():
    """Create a backup of the database."""
    logger.info("Creating database backup...")
    try:
        if create_backup():
            logger.info("Database backup created successfully.")
            return True
        else:
            logger.error("Failed to create database backup.")
            return False
    except Exception as e:
        logger.error(f"Error during database backup: {str(e)}")
        return False


def run_retraining():
    """Run the retraining script."""
    logger.info("Starting scheduled retraining...")

    # First, create a database backup
    backup_success = backup_database()
    if not backup_success:
        logger.warning("Proceeding with retraining despite backup failure.")

    try:
        # Run the retraining script
        start_time = datetime.now()
        logger.info(
            f"Retraining started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        result = subprocess.run(
            [sys.executable, RETRAIN_SCRIPT],
            capture_output=True,
            text=True,
            check=True
        )

        end_time = datetime.now()
        duration = end_time - start_time

        # Log the output
        logger.info(
            f"Retraining completed successfully in {duration.total_seconds()/60:.2f} minutes.")

        # Check if model was updated
        if "Model saved successfully" in result.stdout:
            logger.info("Model was updated with new feedback data.")
            send_email_notification(
                "Model Updated Successfully",
                f"The SKU prediction model was successfully updated with new feedback data.\n\n"
                f"Duration: {duration.total_seconds()/60:.2f} minutes\n\n"
                f"Output Summary:\n{result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}"
            )
        else:
            logger.info("No model update was needed.")

        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"Retraining failed with error code {e.returncode}\n{e.stderr}"
        logger.error(error_msg)
        send_email_notification("Retraining Failed", error_msg)
        return False
    except Exception as e:
        error_msg = f"Unexpected error during retraining: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        send_email_notification("Retraining Failed", error_msg)
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Schedule model retraining with feedback data.')
    parser.add_argument('--now', action='store_true',
                        help='Run retraining immediately')
    parser.add_argument('--daily-time', type=str,
                        default="02:00", help='Daily retraining time (HH:MM)')
    parser.add_argument('--weekly-time', type=str,
                        default="03:00", help='Weekly retraining time (HH:MM)')
    parser.add_argument('--weekly-day', type=str, default="sunday",
                        choices=['monday', 'tuesday', 'wednesday',
                                 'thursday', 'friday', 'saturday', 'sunday'],
                        help='Day for weekly retraining')
    parser.add_argument('--backup-only', action='store_true',
                        help='Only create a database backup without retraining')
    return parser.parse_args()


def main():
    """Schedule and run the retraining process."""
    args = parse_arguments()

    if args.backup_only:
        logger.info("Running in backup-only mode...")
        backup_database()
        return

    logger.info("Starting retraining scheduler...")

    # Schedule retraining to run daily
    schedule.every().day.at(args.daily_time).do(run_retraining)
    logger.info(f"Scheduled daily retraining at {args.daily_time}")

    # Schedule weekly retraining
    weekly_schedule = getattr(schedule.every(), args.weekly_day)
    weekly_schedule.at(args.weekly_time).do(run_retraining)
    logger.info(
        f"Scheduled weekly retraining on {args.weekly_day} at {args.weekly_time}")

    # Also schedule daily database backups
    schedule.every().day.at("00:00").do(backup_database)
    logger.info("Scheduled daily database backups at midnight")

    # Run immediately if requested
    if args.now:
        logger.info("Running initial retraining...")
        run_retraining()

    # Keep the script running
    logger.info("Scheduler is running. Press Ctrl+C to exit.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")
    except Exception as e:
        error_msg = f"Scheduler stopped due to error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        send_email_notification("Scheduler Stopped", error_msg)


if __name__ == "__main__":
    main()
