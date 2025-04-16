import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np
import re

# Paths to model artifacts
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
MODEL_PATH = os.path.join(MODEL_DIR, 'lgbm_sku_predictor.joblib')
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, 'preprocessor.joblib')
LABEL_ENCODER_PATH = os.path.join(PROCESSED_DIR, 'label_encoder.joblib')

# Try to load model and preprocessors at startup, fallback to None if missing
lgbm_model = None
preprocessor = None
label_encoder = None
try:
    lgbm_model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
except Exception as e:
    print(f"[WARN] Model or preprocessors not loaded: {e}\nRunning in dummy mode.")

app = Flask(__name__)

# Text cleaning function (same as in preprocess)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
    else:
        # Dummy mode
        sku_pred = "SKU_DUMMY"
    return jsonify({'sku': sku_pred})

@app.route('/static/<path:filename>')
def static_files(filename):
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    return send_from_directory(static_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
