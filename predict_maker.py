import os
import sys
import joblib
import string

# --- Paths ---
MODEL_DIR = "models/hierarchical"
MAKER_MODEL_PATH = os.path.join(MODEL_DIR, "maker_model.joblib")
MAKER_PREPROCESSORS_PATH = os.path.join(MODEL_DIR, "maker_preprocessors.joblib")

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

def main():
    # Check if model files exist
    if not os.path.exists(MAKER_MODEL_PATH):
        print(f"Maker model not found at {MAKER_MODEL_PATH}")
        return
    
    if not os.path.exists(MAKER_PREPROCESSORS_PATH):
        print(f"Maker preprocessors not found at {MAKER_PREPROCESSORS_PATH}")
        return
    
    # Load maker model and preprocessors
    print("Loading maker model and preprocessors...")
    maker_model = joblib.load(MAKER_MODEL_PATH)
    maker_preprocessors = joblib.load(MAKER_PREPROCESSORS_PATH)
    
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

if __name__ == "__main__":
    main()
