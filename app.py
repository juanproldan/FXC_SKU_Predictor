import os
import joblib
import string
import json
import logging
from datetime import datetime
from flask import Flask, request, render_template, jsonify

# Import feedback database module
from src.feedback_db import save_feedback, get_feedback_stats

# Set up logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
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

# --- Constants ---
MODEL_DIR = "models/renault_neural"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.joblib")

app = Flask(__name__)

# Load model and preprocessing objects
logger.info("Loading model and preprocessing objects...")
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    metadata = joblib.load(METADATA_PATH)

    logger.info(
        f"Model covers {metadata['num_skus']} SKUs with at least {metadata['min_examples_per_sku']} examples each")
    logger.info(f"Test accuracy: {metadata['test_accuracy']:.3f}")
    print(
        f"Model covers {metadata['num_skus']} SKUs with at least {metadata['min_examples_per_sku']} examples each")
    print(f"Test accuracy: {metadata['test_accuracy']:.3f}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise


def preprocess_text(text):
    """Clean and normalize text data."""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def predict_sku(description):
    """Predict the SKU for a given description."""
    # Preprocess the description
    processed_desc = preprocess_text(description)

    # Extract features
    X = vectorizer.transform([processed_desc])

    # Predict SKU
    if hasattr(model, 'predict_proba'):
        # For models that support probability estimates
        probabilities = model.predict_proba(X)[0]
        top_indices = probabilities.argsort()[-5:][::-1]
        top_skus = []
        for idx in top_indices:
            sku = label_encoder.inverse_transform([idx])[0]
            # Convert to float for JSON serialization
            confidence = float(probabilities[idx])
            top_skus.append({"sku": sku, "confidence": confidence})

        # Get the top prediction
        top_idx = probabilities.argmax()
        top_sku = label_encoder.inverse_transform([top_idx])[0]
        top_confidence = float(probabilities[top_idx])
    else:
        # For models that don't support probability estimates
        top_sku = model.predict(X)[0]
        top_sku = label_encoder.inverse_transform([top_sku])[0]
        top_confidence = 1.0
        top_skus = [{"sku": top_sku, "confidence": top_confidence}]

    return {
        "top_sku": top_sku,
        "top_confidence": top_confidence,
        "top_skus": top_skus
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/admin')
def admin():
    """Admin page to view feedback statistics."""
    return render_template('admin.html')


@app.route('/predict', methods=['POST'])
def predict():
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

        # Make prediction
        result = predict_sku(description)

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
    try:
        # Get description from JSON
        data = request.get_json()

        if not data or 'description' not in data:
            return jsonify({"error": "No description provided"}), 400

        description = data['description']

        # Make prediction
        result = predict_sku(description)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/maker_series_model', methods=['GET'])
def get_maker_series_model():
    """Return a hierarchical structure of maker, series, and model."""
    try:
        # For now, just return a simple structure with Renault
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
            }
        }
        return jsonify(data)
    except Exception as e:
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
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
