#!/usr/bin/env python
"""
Script to train a neural network model for SKU prediction with multiple makers.

This script processes the data from the CSV files and trains a neural network model
that can predict SKUs for multiple makers, including Mazda.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_multi_maker_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(
    "C:/Users/juanp/OneDrive/Documents/Python/0_Training/017_Fixacar/maker_csv_detailed")
OUTPUT_DIR = BASE_DIR / "models" / "multi_maker_neural"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters
MIN_EXAMPLES_PER_SKU = 10
MAX_SKUS = 100  # Increased to accommodate multiple makers


def preprocess_text(text):
    """
    Preprocess text for model training.

    Args:
        text: The text to preprocess

    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and extra spaces
    text = ' '.join(text.split())

    return text


def load_data():
    """
    Load and preprocess data from CSV files.

    Returns:
        DataFrame containing the preprocessed data
    """
    logger.info("Loading data from CSV files...")

    # List of makers to include
    makers = ["renault", "mazda", "chevrolet", "ford"]

    all_data = []

    for maker in makers:
        # Try both lowercase and uppercase filenames
        for filename in [f"{maker}_detailed.csv", f"{maker.upper()}_detailed.csv"]:
            file_path = DATA_DIR / filename
            if file_path.exists():
                logger.info(f"Loading data from {file_path}")
                try:
                    # Read CSV file
                    df = pd.read_csv(file_path)

                    # Keep only necessary columns
                    if "maker" in df.columns and "descripcion" in df.columns and "referencia" in df.columns:
                        df = df[["maker", "descripcion", "referencia"]]

                        # Filter out rows with missing values
                        df = df.dropna(subset=["descripcion", "referencia"])

                        # Add to the list
                        all_data.append(df)
                        logger.info(f"Loaded {len(df)} rows from {filename}")
                    else:
                        logger.warning(
                            f"Required columns not found in {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")

    if not all_data:
        logger.error("No data loaded. Check file paths and formats.")
        sys.exit(1)

    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined_data)}")

    # Preprocess descriptions
    logger.info("Preprocessing descriptions...")
    combined_data["processed_description"] = combined_data["descripcion"].apply(
        preprocess_text)

    # Add maker to description for better context
    combined_data["full_description"] = combined_data["maker"] + \
        " " + combined_data["processed_description"]

    return combined_data


def prepare_data_for_training(df):
    """
    Prepare data for model training.

    Args:
        df: DataFrame containing the preprocessed data

    Returns:
        Tuple containing:
            - X_train: Training features
            - X_test: Testing features
            - y_train: Training labels
            - y_test: Testing labels
            - vectorizer: TF-IDF vectorizer
            - label_encoder: Label encoder for SKUs
            - selected_skus: List of selected SKUs
    """
    logger.info("Preparing data for training...")

    # Count occurrences of each SKU
    sku_counts = df["referencia"].value_counts()

    # Select SKUs with at least MIN_EXAMPLES_PER_SKU examples
    selected_skus = sku_counts[sku_counts >=
                               MIN_EXAMPLES_PER_SKU].index.tolist()

    # Limit to MAX_SKUS most common SKUs
    if len(selected_skus) > MAX_SKUS:
        logger.info(f"Limiting to {MAX_SKUS} most common SKUs")
        selected_skus = sku_counts.nlargest(MAX_SKUS).index.tolist()

    logger.info(f"Selected {len(selected_skus)} SKUs for training")

    # Filter data to include only selected SKUs
    filtered_df = df[df["referencia"].isin(selected_skus)]
    logger.info(f"Filtered data contains {len(filtered_df)} rows")

    # Split into features and labels
    X = filtered_df["full_description"]
    y = filtered_df["referencia"]

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2)
    )

    # Create label encoder
    label_encoder = LabelEncoder()

    # Fit and transform
    X_vectorized = vectorizer.fit_transform(X)
    y_encoded = label_encoder.fit_transform(y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Testing set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, vectorizer, label_encoder, selected_skus


def train_model(X_train, y_train):
    """
    Train a neural network model.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained model
    """
    logger.info("Training neural network model...")

    # Create and train the model
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=32,
        learning_rate="adaptive",
        max_iter=200,
        random_state=42,
        verbose=True
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model.

    Args:
        model: Trained model
        X_test: Testing features
        y_test: Testing labels
        label_encoder: Label encoder for SKUs

    Returns:
        Accuracy score
    """
    logger.info("Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.4f}")

    # Print some example predictions
    logger.info("Example predictions:")
    for i in range(min(5, len(y_test))):
        true_sku = label_encoder.inverse_transform([y_test[i]])[0]
        pred_sku = label_encoder.inverse_transform([y_pred[i]])[0]
        logger.info(f"True: {true_sku}, Predicted: {pred_sku}")

    return accuracy


def save_model(model, vectorizer, label_encoder, selected_skus, accuracy):
    """
    Save the trained model and associated objects.

    Args:
        model: Trained model
        vectorizer: TF-IDF vectorizer
        label_encoder: Label encoder for SKUs
        selected_skus: List of selected SKUs
        accuracy: Test accuracy
    """
    logger.info(f"Saving model to {OUTPUT_DIR}...")

    # Save model
    joblib.dump(model, OUTPUT_DIR / "model.joblib")

    # Save vectorizer
    joblib.dump(vectorizer, OUTPUT_DIR / "vectorizer.joblib")

    # Save label encoder
    joblib.dump(label_encoder, OUTPUT_DIR / "label_encoder.joblib")

    # Save metadata
    metadata = {
        "min_examples_per_sku": MIN_EXAMPLES_PER_SKU,
        "max_skus": MAX_SKUS,
        "num_skus": len(selected_skus),
        "skus": selected_skus,
        "test_accuracy": accuracy,
        "last_retrained": datetime.now().isoformat()
    }
    joblib.dump(metadata, OUTPUT_DIR / "metadata.joblib")

    logger.info("Model saved successfully")


def main():
    """Main function to train the model."""
    try:
        logger.info("Starting model training...")

        # Load and preprocess data
        data = load_data()

        # Prepare data for training
        X_train, X_test, y_train, y_test, vectorizer, label_encoder, selected_skus = prepare_data_for_training(
            data)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test, label_encoder)

        # Save model
        save_model(model, vectorizer, label_encoder, selected_skus, accuracy)

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
