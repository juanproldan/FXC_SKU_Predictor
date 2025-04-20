#!/usr/bin/env python
"""
Deployment script for the SKU prediction system.

This script handles the deployment of the SKU prediction system to a production environment.
It includes functions for setting up the environment, installing dependencies, and starting
the application.
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime

# Set up logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f'deploy-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('deploy')


def run_command(command, cwd=None):
    """Run a shell command and log the output.
    
    Args:
        command (str): The command to run.
        cwd (str, optional): The working directory to run the command in.
        
    Returns:
        bool: True if the command succeeded, False otherwise.
    """
    logger.info(f"Running command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True
        )
        logger.info(f"Command output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False


def install_dependencies():
    """Install the required dependencies."""
    logger.info("Installing dependencies...")
    return run_command("pip install -r requirements.txt")


def setup_environment():
    """Set up the environment for deployment."""
    logger.info("Setting up environment...")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'production'
    os.environ['SECRET_KEY'] = 'your-production-secret-key'  # Change this in production
    
    return True


def start_application(host='0.0.0.0', port=5000):
    """Start the application in production mode.
    
    Args:
        host (str): The host to run the application on.
        port (int): The port to run the application on.
        
    Returns:
        bool: True if the application started successfully, False otherwise.
    """
    logger.info(f"Starting application on {host}:{port}...")
    return run_command(f"python run.py web --production --host {host} --port {port}")


def deploy(host='0.0.0.0', port=5000):
    """Deploy the application.
    
    Args:
        host (str): The host to run the application on.
        port (int): The port to run the application on.
        
    Returns:
        bool: True if the deployment succeeded, False otherwise.
    """
    logger.info("Starting deployment...")
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Set up environment
    if not setup_environment():
        logger.error("Failed to set up environment")
        return False
    
    # Start application
    if not start_application(host, port):
        logger.error("Failed to start application")
        return False
    
    logger.info("Deployment completed successfully")
    return True


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Deploy the SKU prediction system')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the application on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the application on')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    success = deploy(host=args.host, port=args.port)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
