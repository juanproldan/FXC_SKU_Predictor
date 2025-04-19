import os
import sys
import time
import schedule
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("retraining_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Path to the retraining script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RETRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "retrain_with_feedback.py")

def run_retraining():
    """Run the retraining script."""
    logger.info("Starting scheduled retraining...")
    
    try:
        # Run the retraining script
        result = subprocess.run(
            [sys.executable, RETRAIN_SCRIPT],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Log the output
        logger.info("Retraining completed successfully.")
        logger.info(f"Output: {result.stdout}")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed with error code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during retraining: {str(e)}")
        return False

def main():
    """Schedule and run the retraining process."""
    logger.info("Starting retraining scheduler...")
    
    # Schedule retraining to run daily at 2 AM
    schedule.every().day.at("02:00").do(run_retraining)
    
    # Also schedule to run weekly on Sunday at 3 AM for a more thorough retraining
    schedule.every().sunday.at("03:00").do(run_retraining)
    
    # Run once at startup
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
        logger.error(f"Scheduler stopped due to error: {str(e)}")

if __name__ == "__main__":
    main()
