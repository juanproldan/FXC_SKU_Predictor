#!/usr/bin/env python
"""
Main entry point for the SKU prediction system.

This script provides a command-line interface for running the SKU prediction
system in various modes.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Set up logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'run.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('run')


def run_web_app(host='127.0.0.1', port=5000, debug=False, production=False):
    """Run the web application.

    Args:
        host (str): The host to run the web application on.
        port (int): The port to run the web application on.
        debug (bool): Whether to run the web application in debug mode.
        production (bool): Whether to run the web application in production mode.
    """
    try:
        from fxc_sku_predictor.web.app import create_app
        app = create_app(production=production)
        logger.info(
            f"Starting web application on {host}:{port} (production={production})")

        if production:
            # In production, we use a WSGI server
            from waitress import serve
            serve(app, host=host, port=port)
        else:
            # In development, we use Flask's built-in server
            app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error running web application: {str(e)}", exc_info=True)
        sys.exit(1)


def run_backup():
    """Run the database backup process."""
    try:
        from fxc_sku_predictor.utils.backup import create_backup, list_backups
        logger.info("Starting database backup process...")
        if create_backup():
            logger.info("Database backup completed successfully")
        else:
            logger.error("Database backup failed")
            sys.exit(1)
        list_backups()
    except Exception as e:
        logger.error(f"Error running database backup: {str(e)}", exc_info=True)
        sys.exit(1)


def run_monitoring(hours=24, notify=False, continuous=False, interval=3600):
    """Run the log monitoring process.

    Args:
        hours (int): Number of hours of logs to check.
        notify (bool): Whether to send email notifications.
        continuous (bool): Whether to run in continuous mode.
        interval (int): Check interval in seconds (for continuous mode).
    """
    try:
        # Create a simple object to hold the arguments
        class Args:
            pass
        args = Args()
        args.hours = hours
        args.notify = notify
        args.continuous = continuous
        args.interval = interval

        from fxc_sku_predictor.utils.monitoring import monitor_logs
        logger.info(f"Starting log monitoring (checking last {hours} hours)")

        if continuous:
            from fxc_sku_predictor.utils.monitoring import main as monitoring_main
            monitoring_main()
        else:
            errors = monitor_logs(args)
            if errors:
                logger.warning(
                    f"Found errors in logs: {sum(len(e) for e in errors.values())} total errors")
                sys.exit(1)
            else:
                logger.info("No errors found in logs")
    except Exception as e:
        logger.error(f"Error running log monitoring: {str(e)}", exc_info=True)
        sys.exit(1)


def run_prediction(description):
    """Run a prediction for a single description.

    Args:
        description (str): The description to predict the SKU for.
    """
    try:
        from fxc_sku_predictor.models.neural_network import predict_sku
        logger.info(f"Predicting SKU for description: '{description}'")
        result = predict_sku(description)

        print(f"\nDescription: {description}")
        print(f"Predicted SKU: {result['top_sku']}")
        print(f"Confidence: {result['top_confidence']:.3f}")
        print("\nTop 5 SKUs:")
        for i, sku_info in enumerate(result['top_skus']):
            print(f"{i+1}. {sku_info['sku']}: {sku_info['confidence']:.3f}")
    except Exception as e:
        logger.error(f"Error running prediction: {str(e)}", exc_info=True)
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='SKU Prediction System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Web application command
    web_parser = subparsers.add_parser('web', help='Run the web application')
    web_parser.add_argument('--host', type=str, default='127.0.0.1',
                            help='Host to run the web application on')
    web_parser.add_argument('--port', type=int, default=5000,
                            help='Port to run the web application on')
    web_parser.add_argument('--debug', action='store_true',
                            help='Run the web application in debug mode')
    web_parser.add_argument('--production', action='store_true',
                            help='Run the web application in production mode')

    # Backup command
    backup_parser = subparsers.add_parser(
        'backup', help='Run the database backup process')

    # Monitoring command
    monitor_parser = subparsers.add_parser(
        'monitor', help='Run the log monitoring process')
    monitor_parser.add_argument(
        '--hours', type=int, default=24, help='Hours of logs to check')
    monitor_parser.add_argument(
        '--notify', action='store_true', help='Send email notifications')
    monitor_parser.add_argument(
        '--continuous', action='store_true', help='Run in continuous monitoring mode')
    monitor_parser.add_argument('--interval', type=int, default=3600,
                                help='Check interval in seconds (for continuous mode)')

    # Prediction command
    predict_parser = subparsers.add_parser(
        'predict', help='Run a prediction for a single description')
    predict_parser.add_argument(
        'description', type=str, help='Description to predict the SKU for')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    if args.command == 'web':
        run_web_app(host=args.host, port=args.port,
                    debug=args.debug, production=args.production)
    elif args.command == 'backup':
        run_backup()
    elif args.command == 'monitor':
        run_monitoring(hours=args.hours, notify=args.notify,
                       continuous=args.continuous, interval=args.interval)
    elif args.command == 'predict':
        run_prediction(args.description)
    else:
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
