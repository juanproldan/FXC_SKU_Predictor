"""
Web application for the SKU prediction system.

This module provides a Flask web application for serving SKU predictions
through a web interface.
"""

import os
import logging
from datetime import datetime
from flask import Flask, request, render_template, jsonify

# Import from our package
from fxc_sku_predictor.models.neural_network import predict_sku, load_model
from fxc_sku_predictor.core.feedback_db import save_feedback, get_feedback_stats

# Set up logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'app.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('app')

# Create Flask application
app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'static'))

# Load model at startup
try:
    model, vectorizer, label_encoder, metadata = load_model()
    logger.info(f"Model loaded successfully with {metadata['num_skus']} SKUs")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model, vectorizer, label_encoder = None, None, None

# --- Routes ---


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/admin')
def admin():
    """Render the admin page."""
    return render_template('admin.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get data from JSON
        data = request.get_json()

        if not data:
            logger.warning("Predict endpoint called with no data")
            return jsonify({"error": "No data provided"}), 400

        description = data.get('description', '')
        maker = data.get('maker', '')
        series = data.get('series', '')
        model_year = data.get('model_year', '')

        logger.info(
            f"Prediction request: maker={maker}, series={series}, model_year={model_year}, description='{description}'")

        if not description:
            logger.warning("Predict endpoint called with empty description")
            return jsonify({"error": "No description provided"}), 400

        # Get maker if provided
        maker = request.form.get('maker')

        # Make prediction
        result = predict_sku(description, maker, model,
                             vectorizer, label_encoder)

        # Add model information
        result['sku'] = result.pop('top_sku')
        result['confidence'] = result.pop('top_confidence')
        result['model_used'] = 'neural_network'

        logger.info(
            f"Prediction result: SKU={result['sku']}, confidence={result['confidence']:.3f}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        # Get description from JSON
        data = request.get_json()

        if not data or 'description' not in data:
            return jsonify({"error": "No description provided"}), 400

        description = data['description']

        # Get maker if provided
        maker = data.get('maker')

        # Make prediction
        result = predict_sku(description, maker, model,
                             vectorizer, label_encoder)

        return jsonify(result)

    except Exception as e:
        logger.error(
            f"Error in API prediction endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/maker_series_model', methods=['GET'])
def get_maker_series_model():
    """Return a hierarchical structure of maker, series, and model."""
    try:
        # Return a structure with multiple makers including Mazda
        data = {
            "RENAULT": {
                "LOGAN": ["2015", "2016", "2017", "2018", "2019", "2020"],
                "SANDERO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                "DUSTER": ["2015", "2016", "2017", "2018", "2019", "2020"],
                "CLIO": ["2010", "2011", "2012", "2013", "2014", "2015"]
            },
            "CHEVROLET": {
                "SPARK": ["2015", "2016", "2017", "2018", "2019", "2020"],
                "AVEO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                "CRUZE": ["2015", "2016", "2017", "2018", "2019", "2020"]
            },
            "FORD": {
                "FIESTA": ["2015", "2016", "2017", "2018", "2019", "2020"],
                "FOCUS": ["2015", "2016", "2017", "2018", "2019", "2020"],
                "ESCAPE": ["2015", "2016", "2017", "2018", "2019", "2020"]
            },
            "MAZDA": {
                "MAZDA 2": ["2015", "2016", "2017", "2018", "2019", "2020"],
                "MAZDA 3": ["2015", "2016", "2017", "2018", "2019", "2020"],
                "MAZDA 6": ["2015", "2016", "2017", "2018", "2019", "2020"],
                "CX-5": ["2015", "2016", "2017", "2018", "2019", "2020"]
            }
        }
        return jsonify(data)
    except Exception as e:
        logger.error(
            f"Error in maker_series_model endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Receive and store user feedback on predictions."""
    try:
        # Get feedback data from JSON
        data = request.get_json()

        if not data:
            logger.warning("Feedback endpoint called with no data")
            return jsonify({"error": "No feedback data provided"}), 400

        required_fields = ['description', 'predicted_sku', 'is_correct']
        for field in required_fields:
            if field not in data:
                logger.warning(
                    f"Feedback endpoint called with missing field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()

        # Log the feedback
        is_correct = data.get('is_correct', False)
        predicted_sku = data.get('predicted_sku', '')
        correct_sku = data.get('correct_sku', 'N/A')
        description = data.get('description', '')

        if is_correct:
            logger.info(
                f"Correct prediction feedback: SKU={predicted_sku}, description='{description}'")
        else:
            logger.info(
                f"Incorrect prediction feedback: Predicted={predicted_sku}, Correct={correct_sku}, description='{description}'")

        # Save feedback to database
        feedback_id = save_feedback(data)
        logger.info(f"Feedback saved with ID: {feedback_id}")

        return jsonify({
            "success": True,
            "message": "Feedback received successfully",
            "feedback_id": feedback_id
        })

    except Exception as e:
        logger.error(f"Error in feedback endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/feedback/stats', methods=['GET'])
def get_feedback_statistics():
    """Get statistics about collected feedback."""
    try:
        stats = get_feedback_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(
            f"Error in feedback stats endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- Main Execution ---


def create_app():
    """Create and configure the Flask application."""
    return app


if __name__ == '__main__':
    app.run(debug=True)
