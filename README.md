# FXC SKU Predictor

A machine learning system for predicting SKUs (Stock Keeping Units) from product descriptions for automotive parts.

## Overview

The FXC SKU Predictor is a system that uses machine learning to predict the correct SKU for a product based on its description. It is designed to help users quickly find the right part number without having to search through catalogs or databases.

The system includes:

- A neural network model trained on product descriptions and their corresponding SKUs
- A web interface for making predictions and collecting user feedback
- A feedback mechanism that allows the model to learn from user corrections
- Automated retraining to continuously improve prediction accuracy
- Monitoring and backup utilities for production deployment

## Features

- **SKU Prediction**: Predict the most likely SKU for a given product description
- **Confidence Scores**: Each prediction includes a confidence score to indicate reliability
- **Alternative Suggestions**: View alternative SKUs when the top prediction is uncertain
- **User Feedback**: Provide feedback on predictions to improve future results
- **Continuous Learning**: The model automatically retrains with user feedback
- **Web Interface**: User-friendly interface for making predictions
- **Admin Dashboard**: Monitor prediction accuracy and user feedback
- **Database Backup**: Automated backup of the feedback database
- **Error Monitoring**: Monitor logs for errors and receive notifications

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/juanproldan/FXC_SKU_Predictor.git
   cd FXC_SKU_Predictor
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up the project structure:
   ```
   mkdir -p logs backups data
   ```

## Usage

### Running the Web Application

To start the web application:

```
python run.py web
```

This will start the Flask application on http://127.0.0.1:5000.

Options:
- `--host`: Host to run the web application on (default: 127.0.0.1)
- `--port`: Port to run the web application on (default: 5000)
- `--debug`: Run the web application in debug mode

### Making Predictions from the Command Line

To make a prediction for a single description:

```
python run.py predict "amortiguador delantero renault logan"
```

### Database Backup

To create a backup of the feedback database:

```
python run.py backup
```

### Log Monitoring

To check logs for errors:

```
python run.py monitor
```

Options:
- `--hours`: Hours of logs to check (default: 24)
- `--notify`: Send email notifications for errors
- `--continuous`: Run in continuous monitoring mode
- `--interval`: Check interval in seconds for continuous mode (default: 3600)

## Project Structure

```
FXC_SKU_Predictor/
├── fxc_sku_predictor/       # Main package
│   ├── core/                # Core functionality
│   │   └── feedback_db.py   # Feedback database module
│   ├── data/                # Data handling
│   ├── models/              # Model definitions
│   │   └── neural_network.py # Neural network model
│   ├── scripts/             # Scripts for various tasks
│   ├── tests/               # Unit tests
│   ├── utils/               # Utility functions
│   │   ├── backup.py        # Database backup utilities
│   │   ├── monitoring.py    # Log monitoring utilities
│   │   └── text_preprocessing.py # Text preprocessing utilities
│   └── web/                 # Web application
│       └── app.py           # Flask application
├── static/                  # Static files for web application
│   └── style.css            # CSS styles
├── templates/               # HTML templates
│   ├── index.html           # Main page
│   └── admin.html           # Admin dashboard
├── data/                    # Data directory
│   └── feedback.db          # Feedback database
├── logs/                    # Log files
├── backups/                 # Database backups
├── models/                  # Trained models
│   └── renault_neural/      # Neural network model for Renault
├── docs/                    # Documentation
├── run.py                   # Main entry point
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Configuration

### Email Notifications

To enable email notifications for error monitoring, update the `EMAIL_CONFIG` in `fxc_sku_predictor/utils/monitoring.py`:

```python
EMAIL_CONFIG = {
    'enabled': True,  # Set to True to enable email notifications
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'your_email@gmail.com',
    'password': 'your_app_password',  # Use app password for Gmail
    'from_email': 'your_email@gmail.com',
    'to_email': 'recipient@example.com',
    'subject_prefix': '[SKU Predictor Monitor]'
}
```

## Development

### Adding New Models

To add a new prediction model:

1. Create a new module in the `fxc_sku_predictor/models/` directory
2. Implement the `predict_sku` function with the same interface as in `neural_network.py`
3. Update the web application to use the new model

### Running Tests

To run the tests:

```
python -m unittest discover -s fxc_sku_predictor/tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped improve this project
- Special thanks to the Fixacar team for their support and feedback
