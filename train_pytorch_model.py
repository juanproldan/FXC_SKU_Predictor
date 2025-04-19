import os
import pandas as pd
import numpy as np
import string
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# --- Constants ---
DATA_PATH = "Fuente_Json_Consolidado/item_training_data.csv"
MIN_EXAMPLES_PER_SKU = 10
MAX_SKUS = 50
MODEL_DIR = "models/renault_pytorch"
os.makedirs(MODEL_DIR, exist_ok=True)

# Neural network parameters
VOCAB_SIZE = 5000  # Maximum number of words in the vocabulary
EMBEDDING_DIM = 100  # Dimension of word embeddings
HIDDEN_DIM = 128    # Number of hidden units
NUM_EPOCHS = 20     # Number of training epochs
BATCH_SIZE = 32     # Batch size for training
LEARNING_RATE = 0.001  # Learning rate

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

class TextDataset(Dataset):
    """Dataset for text classification."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

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

def collate_batch(batch):
    """Custom collate function for the DataLoader."""
    texts, labels = zip(*batch)
    return torch.stack(texts), torch.tensor(labels)

def main():
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    
    # Preprocess text columns
    for col in ["maker", "series", "model", "descripcion"]:
        df[col] = df[col].fillna('').astype(str).apply(preprocess_text)
    
    # Filter to only include Renault (case-insensitive)
    df_renault = df[df["maker"].str.lower() == "renault"].reset_index(drop=True)
    print(f"Filtered to {len(df_renault)} Renault rows")
    
    # Filter SKUs with too few examples
    sku_counts = df_renault["referencia"].value_counts()
    valid_skus = sku_counts[sku_counts >= MIN_EXAMPLES_PER_SKU].index
    print(f"Keeping {len(valid_skus)} out of {len(sku_counts)} SKUs with at least {MIN_EXAMPLES_PER_SKU} examples")
    
    # Limit to top SKUs
    if len(valid_skus) > MAX_SKUS:
        valid_skus = sku_counts.nlargest(MAX_SKUS).index
        print(f"Limiting to top {MAX_SKUS} SKUs")
    
    df_filtered = df_renault[df_renault["referencia"].isin(valid_skus)].reset_index(drop=True)
    print(f"Dataset size after filtering: {len(df_filtered)} rows")
    
    # Encode SKUs
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_filtered["referencia"])
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    
    # Build vocabulary from descriptions
    descriptions = df_filtered["descripcion"].tolist()
    word_counts = Counter()
    for desc in descriptions:
        word_counts.update(desc.split())
    
    # Keep only the most common words
    vocab = ["<PAD>", "<UNK>"] + [word for word, count in word_counts.most_common(VOCAB_SIZE - 2)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Convert descriptions to indices
    def text_to_indices(text, max_len=20):
        words = text.split()
        indices = [word_to_idx.get(word, 1) for word in words[:max_len]]  # 1 is <UNK>
        # Pad to max_len
        indices = indices + [0] * (max_len - len(indices))  # 0 is <PAD>
        return indices
    
    X = [text_to_indices(desc) for desc in descriptions]
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    
    # Build model
    print("Building neural network model...")
    model = TextClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("Training model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    
    best_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_accuracy = correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_accuracy = correct / total
        print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"Saving best model with accuracy: {best_accuracy:.4f}")
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))
    
    print(f"Best test accuracy: {best_accuracy:.4f}")
    
    # Save vocabulary and label encoder
    vocab_path = os.path.join(MODEL_DIR, "vocab.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    metadata_path = os.path.join(MODEL_DIR, "metadata.joblib")
    
    print(f"Saving vocabulary to {vocab_path}")
    joblib.dump(word_to_idx, vocab_path)
    
    print(f"Saving label encoder to {encoder_path}")
    joblib.dump(label_encoder, encoder_path)
    
    # Save metadata
    metadata = {
        'min_examples_per_sku': MIN_EXAMPLES_PER_SKU,
        'max_skus': MAX_SKUS,
        'num_skus': len(valid_skus),
        'skus': valid_skus.tolist(),
        'vocab_size': len(vocab),
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'test_accuracy': best_accuracy
    }
    
    print(f"Saving metadata to {metadata_path}")
    joblib.dump(metadata, metadata_path)
    
    # Print some example SKUs
    print("\nExample SKUs in the model:")
    for sku in valid_skus[:10]:
        examples = df_filtered[df_filtered["referencia"] == sku]["descripcion"].values[:2]
        print(f"  {sku}: {examples}")

if __name__ == "__main__":
    main()
