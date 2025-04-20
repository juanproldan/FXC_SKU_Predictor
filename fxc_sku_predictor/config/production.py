"""
Production configuration for the SKU prediction system.
"""

import os
import logging
from datetime import datetime

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Logs directory
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Log file with timestamp
LOG_FILE = os.path.join(LOG_DIR, f'app-{datetime.now().strftime("%Y%m%d")}.log')

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOG_FILE,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10,
            'formatter': 'standard',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True
        },
        'werkzeug': {
            'handlers': ['file'],
            'level': 'WARNING',
            'propagate': False
        },
    }
}

# Flask application settings
FLASK_ENV = 'production'
DEBUG = False
TESTING = False
SECRET_KEY = os.environ.get('SECRET_KEY', 'generate-a-secure-key-for-production')

# Database settings
FEEDBACK_DB_PATH = os.path.join(DATA_DIR, 'feedback.db')

# Model settings
MODEL_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'neural_network_model.pkl')
