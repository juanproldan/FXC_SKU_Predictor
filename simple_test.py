import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
MODEL_DIR = "models/hierarchical"
MAKER_MODEL_PATH = os.path.join(MODEL_DIR, "maker_model.joblib")
MAKER_PREPROCESSORS_PATH = os.path.join(MODEL_DIR, "maker_preprocessors.joblib")

def preprocess_text(text):
    """Clean and normalize text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def main():
    # Check if model files exist
    if not os.path.exists(MAKER_MODEL_PATH):
        print(f"Maker model not found at {MAKER_MODEL_PATH}")
        return
    
    if not os.path.exists(MAKER_PREPROCESSORS_PATH):
        print(f"Maker preprocessors not found at {MAKER_PREPROCESSORS_PATH}")
        return
    
    # Load maker model and preprocessors
    try:
        print("Loading maker model...")
        maker_model = joblib.load(MAKER_MODEL_PATH)
        print("Loading maker preprocessors...")
        maker_preprocessors = joblib.load(MAKER_PREPROCESSORS_PATH)
        
        # Get the text preprocessor
        text_preprocessor = maker_preprocessors['text_preprocessor']
        label_encoder = maker_preprocessors['label_encoder']
        
        # Test with a sample description
        description = "amortiguador delantero renault logan 2015"
        processed_description = preprocess_text(description)
        
        print(f"Original description: {description}")
        print(f"Processed description: {processed_description}")
        
        # Transform the description
        X_text = text_preprocessor.transform([processed_description]).toarray()
        print(f"Transformed shape: {X_text.shape}")
        
        # Predict maker
        maker_probs = maker_model.predict_proba(X_text)[0]
        maker_idx = maker_probs.argmax()
        maker_confidence = maker_probs[maker_idx]
        maker_name = label_encoder.inverse_transform([maker_idx])[0]
        
        print(f"Predicted maker: {maker_name} (confidence: {maker_confidence:.3f})")
        
        # Print top 3 makers
        top_indices = maker_probs.argsort()[-3:][::-1]
        print("Top 3 makers:")
        for idx in top_indices:
            name = label_encoder.inverse_transform([idx])[0]
            confidence = maker_probs[idx]
            print(f"  {name}: {confidence:.3f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
