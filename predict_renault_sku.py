import os
import sys
import joblib
import string

# --- Paths ---
MODEL_DIR = "models/renault_simple"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
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

def predict_sku(description, model, vectorizer):
    """Predict the SKU for a given description."""
    # Preprocess the description
    processed_desc = preprocess_text(description)
    
    # Extract features
    X = vectorizer.transform([processed_desc]).toarray()
    
    # Predict SKU
    sku_probs = model.predict_proba(X)[0]
    sku_idx = sku_probs.argmax()
    sku_confidence = sku_probs[sku_idx]
    sku = model.classes_[sku_idx]
    
    # Get top 5 SKUs
    top_indices = sku_probs.argsort()[-5:][::-1]
    top_skus = []
    for idx in top_indices:
        sku_name = model.classes_[idx]
        confidence = sku_probs[idx]
        top_skus.append((sku_name, confidence))
    
    return sku, sku_confidence, top_skus

def main():
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
    
    if not os.path.exists(VECTORIZER_PATH):
        print(f"Vectorizer not found at {VECTORIZER_PATH}")
        return
    
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata not found at {METADATA_PATH}")
        return
    
    # Load model and vectorizer
    print("Loading model and vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
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
    sku, sku_confidence, top_skus = predict_sku(description, model, vectorizer)
    
    print(f"Predicted SKU: {sku} (confidence: {sku_confidence:.3f})")
    print("Top 5 SKUs:")
    for name, confidence in top_skus:
        print(f"  {name}: {confidence:.3f}")

if __name__ == "__main__":
    main()
