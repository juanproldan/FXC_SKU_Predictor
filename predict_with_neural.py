import os
import sys
import joblib
import string
import numpy as np

# --- Paths ---
MODEL_DIR = "models/renault_neural"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.joblib")

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

def predict_sku(description, model, vectorizer, label_encoder):
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
            confidence = probabilities[idx]
            top_skus.append((sku, confidence))
        
        # Get the top prediction
        top_idx = probabilities.argmax()
        top_sku = label_encoder.inverse_transform([top_idx])[0]
        top_confidence = probabilities[top_idx]
    else:
        # For models that don't support probability estimates
        top_sku = model.predict(X)[0]
        top_sku = label_encoder.inverse_transform([top_sku])[0]
        top_confidence = 1.0
        top_skus = [(top_sku, top_confidence)]
    
    return top_sku, top_confidence, top_skus

def main():
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
    
    if not os.path.exists(VECTORIZER_PATH):
        print(f"Vectorizer not found at {VECTORIZER_PATH}")
        return
    
    if not os.path.exists(ENCODER_PATH):
        print(f"Label encoder not found at {ENCODER_PATH}")
        return
    
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata not found at {METADATA_PATH}")
        return
    
    # Load model and preprocessing objects
    print("Loading model and preprocessing objects...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    metadata = joblib.load(METADATA_PATH)
    
    print(f"Model covers {metadata['num_skus']} SKUs with at least {metadata['min_examples_per_sku']} examples each")
    print(f"Test accuracy: {metadata['test_accuracy']:.3f}")
    
    # Get description from command line or prompt user
    if len(sys.argv) > 1:
        description = " ".join(sys.argv[1:])
    else:
        print("\nEnter a product description:")
        description = input("> ")
    
    # Predict SKU
    print(f"\nPredicting SKU for: {description}")
    sku, confidence, top_skus = predict_sku(description, model, vectorizer, label_encoder)
    
    print(f"Predicted SKU: {sku} (confidence: {confidence:.3f})")
    print("Top SKUs:")
    for name, conf in top_skus:
        print(f"  {name}: {conf:.3f}")

if __name__ == "__main__":
    main()
