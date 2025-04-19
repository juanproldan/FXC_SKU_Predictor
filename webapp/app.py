import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np
import re
import string

# Paths to model artifacts
MODEL_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'models')
PROCESSED_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')

# Original model paths
LGBM_MODEL_PATH = os.path.join(MODEL_DIR, 'lgbm_sku_predictor.joblib')
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, 'preprocessor.joblib')
LABEL_ENCODER_PATH = os.path.join(PROCESSED_DIR, 'label_encoder.joblib')

# Neural network model paths
NEURAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'models', 'renault_neural')
NEURAL_MODEL_PATH = os.path.join(NEURAL_MODEL_DIR, 'model.joblib')
VECTORIZER_PATH = os.path.join(NEURAL_MODEL_DIR, 'vectorizer.joblib')
NEURAL_ENCODER_PATH = os.path.join(NEURAL_MODEL_DIR, 'label_encoder.joblib')
METADATA_PATH = os.path.join(NEURAL_MODEL_DIR, 'metadata.joblib')

# Try to load models and preprocessors at startup, fallback to None if missing
lgbm_model = None
preprocessor = None
label_encoder = None
neural_model = None
vectorizer = None
neural_encoder = None
metadata = None

# Load original model
try:
    lgbm_model = joblib.load(LGBM_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("Original model loaded successfully")
except Exception as e:
    print(f"[WARN] Original model or preprocessors not loaded: {e}")

# Load neural network model
try:
    neural_model = joblib.load(NEURAL_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    neural_encoder = joblib.load(NEURAL_ENCODER_PATH)
    metadata = joblib.load(METADATA_PATH)
    print(f"Neural network model loaded successfully")
    print(
        f"Model covers {metadata['num_skus']} SKUs with at least {metadata['min_examples_per_sku']} examples each")
    print(f"Test accuracy: {metadata['test_accuracy']:.3f}")
except Exception as e:
    print(f"[WARN] Neural network model or preprocessors not loaded: {e}")

app = Flask(__name__)

# Text cleaning functions


def clean_text(text):
    """Original text cleaning function (for LGBM model)"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text):
    """Text preprocessing for neural network model"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def predict_sku_neural(description):
    """Predict the SKU for a given description using the neural network model."""
    if not neural_model or not vectorizer or not neural_encoder:
        return None, None, None

    # Preprocess the description
    processed_desc = preprocess_text(description)

    # Extract features
    X = vectorizer.transform([processed_desc])

    # Predict SKU
    if hasattr(neural_model, 'predict_proba'):
        # For models that support probability estimates
        probabilities = neural_model.predict_proba(X)[0]
        top_indices = probabilities.argsort()[-5:][::-1]
        top_skus = []
        for idx in top_indices:
            sku = neural_encoder.inverse_transform([idx])[0]
            # Convert to float for JSON serialization
            confidence = float(probabilities[idx])
            top_skus.append({"sku": sku, "confidence": confidence})

        # Get the top prediction
        top_idx = probabilities.argmax()
        top_sku = neural_encoder.inverse_transform([top_idx])[0]
        top_confidence = float(probabilities[top_idx])
    else:
        # For models that don't support probability estimates
        top_sku = neural_model.predict(X)[0]
        top_sku = neural_encoder.inverse_transform([top_sku])[0]
        top_confidence = 1.0
        top_skus = [{"sku": top_sku, "confidence": top_confidence}]

    return top_sku, top_confidence, top_skus


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    description = data.get('description', '')
    maker = data.get('maker', 'missing')
    series = data.get('series', 'missing')
    model_year = data.get('model_year', 'missing')

    # Initialize response
    response = {}

    # Try neural network model first (for Renault)
    if neural_model and vectorizer and neural_encoder and maker.lower() == 'renault':
        top_sku, top_confidence, top_skus = predict_sku_neural(description)
        if top_sku:
            response = {
                'sku': top_sku,
                'confidence': top_confidence,
                'top_skus': top_skus,
                'model_used': 'neural_network'
            }
            return jsonify(response)

    # Fall back to original model or dummy mode
    # Clean and prepare input
    description_cleaned = clean_text(description)
    X_input = {
        'description_cleaned': [description_cleaned],
        'maker': [maker],
        'series': [series],
        'model_year': [model_year]
    }
    if lgbm_model and preprocessor and label_encoder:
        # Transform input
        X_transformed = preprocessor.transform(X_input)
        # Predict
        y_pred = lgbm_model.predict(X_transformed)
        sku_pred = label_encoder.inverse_transform(y_pred)[0]
        response = {
            'sku': sku_pred,
            'confidence': 0.0,  # Original model doesn't provide confidence
            'top_skus': [{'sku': sku_pred, 'confidence': 1.0}],
            'model_used': 'lgbm'
        }
    else:
        # Dummy mode
        response = {
            'sku': "SKU_DUMMY",
            'confidence': 0.0,
            'top_skus': [{'sku': "SKU_DUMMY", 'confidence': 1.0}],
            'model_used': 'dummy'
        }

    return jsonify(response)


@app.route('/static/<path:filename>')
def static_files(filename):
    static_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'static')
    return send_from_directory(static_dir, filename)


@app.route('/api/maker_series_model')
def maker_series_model():
    # Serve the generated JSON file
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'Fuente_Json_Consolidado', 'maker_series_model.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = f.read()
        return app.response_class(data, mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
