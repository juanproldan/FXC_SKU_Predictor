import os
import pandas as pd
import numpy as np
import string
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Constants ---
DATA_PATH = "Fuente_Json_Consolidado/item_training_data.csv"
MIN_EXAMPLES_PER_SKU = 10
MAX_SKUS = 50
MODEL_DIR = "models/renault_neural"
os.makedirs(MODEL_DIR, exist_ok=True)

# Neural network parameters
MAX_WORDS = 10000  # Maximum number of words in the vocabulary
MAX_LENGTH = 20    # Maximum length of each description
EMBEDDING_DIM = 100  # Dimension of word embeddings
LSTM_UNITS = 128    # Number of LSTM units

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
    
    # Tokenize descriptions
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_filtered["descripcion"])
    
    # Convert descriptions to sequences
    sequences = tokenizer.texts_to_sequences(df_filtered["descripcion"])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build model
    print("Building neural network model...")
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LENGTH),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
        Bidirectional(LSTM(LSTM_UNITS)),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.3f}")
    
    # Save model and tokenizer
    model_path = os.path.join(MODEL_DIR, "model.h5")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    metadata_path = os.path.join(MODEL_DIR, "metadata.joblib")
    
    print(f"Saving model to {model_path}")
    model.save(model_path)
    
    print(f"Saving tokenizer to {tokenizer_path}")
    joblib.dump(tokenizer, tokenizer_path)
    
    print(f"Saving label encoder to {encoder_path}")
    joblib.dump(label_encoder, encoder_path)
    
    # Save metadata
    metadata = {
        'min_examples_per_sku': MIN_EXAMPLES_PER_SKU,
        'max_skus': MAX_SKUS,
        'num_skus': len(valid_skus),
        'skus': valid_skus.tolist(),
        'max_words': MAX_WORDS,
        'max_length': MAX_LENGTH,
        'embedding_dim': EMBEDDING_DIM,
        'lstm_units': LSTM_UNITS,
        'test_accuracy': accuracy
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
