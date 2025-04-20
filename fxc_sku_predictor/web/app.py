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
    # Make these variables global so they can be accessed in the routes
    global model, vectorizer, label_encoder, metadata
    model, vectorizer, label_encoder, metadata = load_model()
    logger.info(f"Model loaded successfully with {metadata['num_skus']} SKUs")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model, vectorizer, label_encoder, metadata = None, None, None, None

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
        model_family = data.get('model_family', '')
        series = data.get('series', '')
        model_year = data.get('model_year', '')

        logger.info(
            f"Prediction request: maker={maker}, model_family={model_family}, series={series}, model_year={model_year}, description='{description}'")

        if not description:
            logger.warning("Predict endpoint called with empty description")
            return jsonify({"error": "No description provided"}), 400

        # Use the maker from the JSON data
        # No need to get it from form as we already have it

        # Make prediction
        # Include model_family in the description for better context
        full_description = f"{description} {maker} {model_family} {series} {model_year}".strip(
        )
        # Use the global model, vectorizer, and label_encoder
        result = predict_sku(full_description, maker)

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

        # Get parameters if provided
        maker = data.get('maker', '')
        model_family = data.get('model_family', '')
        series = data.get('series', '')
        model_year = data.get('model_year', '')

        # Make prediction
        # Include model_family in the description for better context
        full_description = f"{description} {maker} {model_family} {series} {model_year}".strip(
        )
        # Use the global model, vectorizer, and label_encoder
        result = predict_sku(full_description, maker)

        return jsonify(result)

    except Exception as e:
        logger.error(
            f"Error in API prediction endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/maker_series_model', methods=['GET'])
def get_maker_series_model():
    """Return a hierarchical structure of maker, series, and model."""
    try:
        # Return a structure with multiple makers including Mazda with model families
        data = {
            "RENAULT": {
                "Duster": {
                    "DUSTER": ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
                    "DUSTER OROCH": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Oroch": {
                    "OROCH": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Clio": {
                    "CLIO": ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Sandero": {
                    "SANDERO": ["2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
                    "SANDERO STEPWAY": ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Captur": {
                    "CAPTUR": ["2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["2016", "2017", "2018", "2019", "2020"]
                },
                "Logan": {
                    "LOGAN": ["2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Alaskan": {
                    "ALASKAN": ["2017", "2018", "2019", "2020"],
                    "STANDARD": ["2017", "2018", "2019", "2020"]
                },
                "Koleos": {
                    "KOLEOS": ["2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Fluence": {
                    "FLUENCE": ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"],
                    "STANDARD": ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]
                },
                "Kwid": {
                    "KWID": ["2017", "2018", "2019", "2020"],
                    "STANDARD": ["2017", "2018", "2019", "2020"]
                },
                "Twingo": {
                    "TWINGO": ["1995", "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012"],
                    "STANDARD": ["1995", "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012"]
                },
                "Megane": {
                    "MEGANE": ["1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Renault 9": {
                    "RENAULT 9": ["1982", "1983", "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995"],
                    "STANDARD": ["1982", "1983", "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995"]
                },
                "Renault 4": {
                    "RENAULT 4": ["1961", "1962", "1963", "1964", "1965", "1966", "1967", "1968", "1969", "1970", "1971", "1972", "1973", "1974", "1975", "1976", "1977", "1978", "1979", "1980", "1981", "1982", "1983", "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992"],
                    "STANDARD": ["1961", "1962", "1963", "1964", "1965", "1966", "1967", "1968", "1969", "1970", "1971", "1972", "1973", "1974", "1975", "1976", "1977", "1978", "1979", "1980", "1981", "1982", "1983", "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992"]
                },
                "Symbol": {
                    "SYMBOL": ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Kangoo": {
                    "KANGOO": ["1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Laguna": {
                    "LAGUNA": ["1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015"],
                    "STANDARD": ["1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015"]
                },
                "Scenic": {
                    "SCENIC": ["1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Renault 19": {
                    "RENAULT 19": ["1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000"],
                    "STANDARD": ["1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000"]
                },
                "Zoe": {
                    "ZOE": ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
                }
            },
            "CHEVROLET": {
                "SPARK": {
                    "STANDARD": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "AVEO": {
                    "STANDARD": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "CRUZE": {
                    "STANDARD": ["2015", "2016", "2017", "2018", "2019", "2020"]
                }
            },
            "FORD": {
                "FIESTA": {
                    "STANDARD": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "FOCUS": {
                    "STANDARD": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "ESCAPE": {
                    "STANDARD": ["2015", "2016", "2017", "2018", "2019", "2020"]
                }
            },
            "MAZDA": {
                "Mazda 2": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "PRIME": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "TOURING PLUS": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING LX": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "SPORT": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "CARBON EDITION": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "100TH ANNIVERSARY": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING PLUS": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING SPOR": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "I TOURING PLUS": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING LX S": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING XL": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "TOURING+": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda 3": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "PRIME-LINE": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "SKYACTIV": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "SPORT": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "CARBON EDITION": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING LX": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "I GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "I TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "S GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "SKYACTIV R": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "STYLE": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "I": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "ENTRY": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "MID": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "S": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "I SPORT GRAND TOUR": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda 5": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda 6": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "ALL NEW": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda 323": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "DI": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "EXCLUSIVE ACTIVE": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "LX": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "COMPAKT": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda 626": {
                    "STANDARD": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda B2000": {
                    "STANDARD": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda BT-50": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "PROFESSIONAL": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda CX-3": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "I GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "I SPORT": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda CX-30": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING LX": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "I GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "PRIME": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda CX-5": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING LX": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "HIGH": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "MID": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "PRIME-LINE": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "LUXURY": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda CX-50": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING LX": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda CX-7": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda CX-9": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING LX": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "GRAND TOURING SIGN": ["2015", "2016", "2017", "2018", "2019", "2020"]
                },
                "Mazda Demio": {
                    "BASICO": ["2015", "2016", "2017", "2018", "2019", "2020"],
                    "STANDARD": ["2015", "2016", "2017", "2018", "2019", "2020"]
                }
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
