import os
import sys
import joblib
import numpy as np
import string

# --- Paths ---
MODEL_DIR = "models/hierarchical"
MAKER_MODEL_PATH = os.path.join(MODEL_DIR, "maker_model.joblib")
MAKER_PREPROCESSORS_PATH = os.path.join(MODEL_DIR, "maker_preprocessors.joblib")
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

def predict_maker(description, maker_model, maker_preprocessors):
    """Predict the maker (brand) for a given description."""
    # Preprocess the description
    processed_desc = preprocess_text(description)
    
    # Extract features
    text_preprocessor = maker_preprocessors['text_preprocessor']
    X_text = text_preprocessor.transform([processed_desc]).toarray()
    
    # Predict maker
    maker_probs = maker_model.predict_proba(X_text)[0]
    maker_idx = maker_probs.argmax()
    maker_confidence = maker_probs[maker_idx]
    maker_name = maker_preprocessors['label_encoder'].inverse_transform([maker_idx])[0]
    
    # Get top 3 makers
    top_indices = maker_probs.argsort()[-3:][::-1]
    top_makers = []
    for idx in top_indices:
        name = maker_preprocessors['label_encoder'].inverse_transform([idx])[0]
        confidence = maker_probs[idx]
        top_makers.append((name, confidence))
    
    return maker_name, maker_confidence, top_makers

def predict_sku(description, maker, sku_model, sku_preprocessors, sku_encoder):
    """Predict the SKU for a given description and maker."""
    # Preprocess the description
    processed_desc = preprocess_text(description)
    
    # Extract features
    cat_preprocessor = sku_preprocessors['cat_preprocessor']
    text_preprocessor = sku_preprocessors['text_preprocessor']
    
    # For simplicity, we'll use empty values for series and model
    # In a real application, you would extract these from the description or ask the user
    X_cat = cat_preprocessor.transform([["", ""]])
    X_text = text_preprocessor.transform([processed_desc]).toarray()
    X = np.hstack([X_cat, X_text])
    
    # Predict SKU
    sku_probs = sku_model.predict_proba(X)[0]
    sku_idx = sku_probs.argmax()
    sku_confidence = sku_probs[sku_idx]
    sku = sku_encoder.inverse_transform([sku_idx])[0]
    
    # Get top 3 SKUs
    top_indices = sku_probs.argsort()[-3:][::-1]
    top_skus = []
    for idx in top_indices:
        sku_name = sku_encoder.inverse_transform([idx])[0]
        confidence = sku_probs[idx]
        top_skus.append((sku_name, confidence))
    
    return sku, sku_confidence, top_skus

def main():
    # Check if model files exist
    if not os.path.exists(MAKER_MODEL_PATH):
        print(f"Maker model not found at {MAKER_MODEL_PATH}")
        return
    
    if not os.path.exists(MAKER_PREPROCESSORS_PATH):
        print(f"Maker preprocessors not found at {MAKER_PREPROCESSORS_PATH}")
        return
    
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata not found at {METADATA_PATH}")
        return
    
    # Load maker model and preprocessors
    print("Loading maker model and preprocessors...")
    maker_model = joblib.load(MAKER_MODEL_PATH)
    maker_preprocessors = joblib.load(MAKER_PREPROCESSORS_PATH)
    metadata = joblib.load(METADATA_PATH)
    
    # Get description from command line or prompt user
    if len(sys.argv) > 1:
        description = " ".join(sys.argv[1:])
    else:
        print("Enter a product description:")
        description = input("> ")
    
    # Predict maker
    print(f"\nPredicting maker for: {description}")
    maker, maker_confidence, top_makers = predict_maker(description, maker_model, maker_preprocessors)
    
    print(f"Predicted maker: {maker} (confidence: {maker_confidence:.3f})")
    print("Top 3 makers:")
    for name, confidence in top_makers:
        print(f"  {name}: {confidence:.3f}")
    
    # Check if we have a model for this maker
    maker_dir = os.path.join(MODEL_DIR, maker.lower().replace(' ', '_'))
    sku_model_path = os.path.join(maker_dir, "sku_model.joblib")
    sku_preprocessors_path = os.path.join(maker_dir, "sku_preprocessors.joblib")
    sku_encoder_path = os.path.join(maker_dir, "sku_encoder.joblib")
    
    if not os.path.exists(sku_model_path):
        print(f"\nNo SKU model found for {maker}")
        return
    
    # Load SKU model and preprocessors
    print(f"\nLoading SKU model for {maker}...")
    sku_model = joblib.load(sku_model_path)
    sku_preprocessors = joblib.load(sku_preprocessors_path)
    sku_encoder = joblib.load(sku_encoder_path)
    
    # Predict SKU
    print(f"Predicting SKU for: {description}")
    sku, sku_confidence, top_skus = predict_sku(description, maker, sku_model, sku_preprocessors, sku_encoder)
    
    print(f"Predicted SKU: {sku} (confidence: {sku_confidence:.3f})")
    print("Top 3 SKUs:")
    for name, confidence in top_skus:
        print(f"  {name}: {confidence:.3f}")

if __name__ == "__main__":
    main()
