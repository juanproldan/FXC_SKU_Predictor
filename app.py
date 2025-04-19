import os
import joblib
import string
import json
from datetime import datetime
from flask import Flask, request, render_template, jsonify

# Import feedback database module
from src.feedback_db import save_feedback, get_feedback_stats

# --- Constants ---
MODEL_DIR = "models/renault_neural"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.joblib")

app = Flask(__name__)

# Load model and preprocessing objects
print("Loading model and preprocessing objects...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
metadata = joblib.load(METADATA_PATH)

print(
    f"Model covers {metadata['num_skus']} SKUs with at least {metadata['min_examples_per_sku']} examples each")
print(f"Test accuracy: {metadata['test_accuracy']:.3f}")


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
            return jsonify({"error": "No data provided"}), 400

        description = data.get('description', '')
        maker = data.get('maker', '')

        if not description:
            return jsonify({"error": "No description provided"}), 400

        # Make prediction
        result = predict_sku(description)

        # Add model information
        result['sku'] = result.pop('top_sku')
        result['confidence'] = result.pop('top_confidence')
        result['model_used'] = 'neural_network'

        return jsonify(result)

    except Exception as e:
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
            return jsonify({"error": "No feedback data provided"}), 400

        required_fields = ['description', 'predicted_sku', 'is_correct']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()

        # Save feedback to database
        feedback_id = save_feedback(data)

        return jsonify({
            "success": True,
            "message": "Feedback received successfully",
            "feedback_id": feedback_id
        })

    except Exception as e:
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
