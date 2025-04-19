import os
import sys
import joblib
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Paths ---
MODEL_DIR = "models/renault_neural"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.joblib")
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

def predict_sku(description, model, tokenizer, label_encoder, max_length):
    """Predict the SKU for a given description using the neural network model."""
    # Preprocess the description
    processed_desc = preprocess_text(description)
    
    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([processed_desc])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Predict
    predictions = model.predict(padded_sequence)[0]
    
    # Get top 5 predictions
    top_indices = predictions.argsort()[-5:][::-1]
    top_skus = []
    for idx in top_indices:
        sku = label_encoder.inverse_transform([idx])[0]
        confidence = predictions[idx]
        top_skus.append((sku, confidence))
    
    # Get the top prediction
    top_idx = predictions.argmax()
    top_sku = label_encoder.inverse_transform([top_idx])[0]
    top_confidence = predictions[top_idx]
    
    return top_sku, top_confidence, top_skus

def main():
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
    
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer not found at {TOKENIZER_PATH}")
        return
    
    if not os.path.exists(ENCODER_PATH):
        print(f"Label encoder not found at {ENCODER_PATH}")
        return
    
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata not found at {METADATA_PATH}")
        return
    
    # Load model and preprocessing objects
    print("Loading model and preprocessing objects...")
    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    metadata = joblib.load(METADATA_PATH)
    
    max_length = metadata['max_length']
    
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
    sku, confidence, top_skus = predict_sku(description, model, tokenizer, label_encoder, max_length)
    
    print(f"Predicted SKU: {sku} (confidence: {confidence:.3f})")
    print("Top 5 SKUs:")
    for name, conf in top_skus:
        print(f"  {name}: {conf:.3f}")

if __name__ == "__main__":
    main()
