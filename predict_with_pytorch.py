import os
import sys
import joblib
import string
import torch
import torch.nn as nn
import numpy as np

# --- Paths ---
MODEL_DIR = "models/renault_pytorch"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.joblib")
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

class TextClassifier(nn.Module):
    """Neural network for text classification."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Sum the embeddings for all words in the text (simple bag of words approach)
        x = self.embedding(x).sum(dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def predict_sku(description, model, word_to_idx, label_encoder, device):
    """Predict the SKU for a given description using the PyTorch model."""
    # Preprocess the description
    processed_desc = preprocess_text(description)
    
    # Convert to indices
    max_len = 20
    words = processed_desc.split()
    indices = [word_to_idx.get(word, 1) for word in words[:max_len]]  # 1 is <UNK>
    # Pad to max_len
    indices = indices + [0] * (max_len - len(indices))  # 0 is <PAD>
    
    # Convert to tensor
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Get top 5 predictions
    top_indices = torch.argsort(probabilities, descending=True)[:5].cpu().numpy()
    top_skus = []
    for idx in top_indices:
        sku = label_encoder.inverse_transform([idx])[0]
        confidence = probabilities[idx].item()
        top_skus.append((sku, confidence))
    
    # Get the top prediction
    top_idx = probabilities.argmax().item()
    top_sku = label_encoder.inverse_transform([top_idx])[0]
    top_confidence = probabilities[top_idx].item()
    
    return top_sku, top_confidence, top_skus

def main():
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
    
    if not os.path.exists(VOCAB_PATH):
        print(f"Vocabulary not found at {VOCAB_PATH}")
        return
    
    if not os.path.exists(ENCODER_PATH):
        print(f"Label encoder not found at {ENCODER_PATH}")
        return
    
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata not found at {METADATA_PATH}")
        return
    
    # Load model and preprocessing objects
    print("Loading model and preprocessing objects...")
    word_to_idx = joblib.load(VOCAB_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    metadata = joblib.load(METADATA_PATH)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = TextClassifier(
        metadata['vocab_size'],
        metadata['embedding_dim'],
        metadata['hidden_dim'],
        metadata['num_skus']
    )
    
    # Load model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    
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
    sku, confidence, top_skus = predict_sku(description, model, word_to_idx, label_encoder, device)
    
    print(f"Predicted SKU: {sku} (confidence: {confidence:.3f})")
    print("Top 5 SKUs:")
    for name, conf in top_skus:
        print(f"  {name}: {conf:.3f}")

if __name__ == "__main__":
    main()
