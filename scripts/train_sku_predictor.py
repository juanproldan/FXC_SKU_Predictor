import os
import sys
import traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from typing import Tuple, Dict
import logging

# --- Logging setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Paths ---
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(
    __file__)), 'Fuente_Json_Consolidado', 'item_training_data.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
PROCESSED_DIR = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'data', 'processed')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'lgbm_sku_predictor.joblib')
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, 'preprocessor.joblib')
LABEL_ENCODER_PATH = os.path.join(PROCESSED_DIR, 'label_encoder.joblib')


def preprocess_text(text: str) -> str:
    """Basic text preprocessing."""
    if not isinstance(text, str):
        return ''
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def prepare_data(df: pd.DataFrame, min_class_count: int = 5, max_classes: int = 100, sample_size: int = 5000) -> Tuple[np.ndarray, np.ndarray, Dict, LabelEncoder]:
    """Prepare features and target with improved preprocessing.

    Args:
        df: Input DataFrame
        min_class_count: Minimum number of examples per class to keep
        max_classes: Maximum number of classes to keep (most frequent)
        sample_size: Maximum number of rows to use for training

    Returns:
        X_all: Feature matrix
        y: Target vector
        preprocessors: Dictionary of preprocessing objects
        label_encoder: Label encoder for target
    """
    logger.info("Starting data preparation...")

    # Take a smaller sample to speed up training
    if len(df) > sample_size:
        logger.info(
            f"Taking a random sample of {sample_size} rows from {len(df)} total rows")
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    # Preprocess text columns
    for col in ["maker", "series", "model", "descripcion"]:
        df[col] = df[col].fillna('').astype(str).apply(preprocess_text)

    # Filter classes with too few examples
    if min_class_count > 1:
        logger.info(
            f"Filtering classes with fewer than {min_class_count} examples")
        class_counts = df["referencia"].value_counts()
        valid_classes = class_counts[class_counts >= min_class_count].index
        logger.info(
            f"Keeping {len(valid_classes)} out of {len(class_counts)} classes")

        # Limit to max_classes most frequent classes
        if len(valid_classes) > max_classes:
            logger.info(f"Limiting to {max_classes} most frequent classes")
            valid_classes = class_counts.nlargest(max_classes).index

        df = df[df["referencia"].isin(valid_classes)].reset_index(drop=True)
        logger.info(f"Dataset size after filtering: {len(df)} rows")

    # Categorical features - simpler preprocessing
    cat_preprocessor = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False,
        min_frequency=2  # Reduced threshold
    )
    X_cat = cat_preprocessor.fit_transform(df[["maker", "series", "model"]])
    logger.info(f"Categorical features shape: {X_cat.shape}")

    # Text features with simpler TF-IDF
    text_preprocessor = TfidfVectorizer(
        max_features=500,           # Reduced features
        min_df=2,                   # Reduced threshold
        max_df=0.9,                 # Ignore very common terms
        ngram_range=(1, 1),         # Only unigrams for simplicity
        strip_accents='unicode',
        analyzer='word'
    )
    X_text = text_preprocessor.fit_transform(df["descripcion"]).toarray()
    logger.info(f"Text features shape: {X_text.shape}")

    # Combine features
    X_all = np.hstack([X_cat, X_text])
    logger.info(f"Combined features shape: {X_all.shape}")

    # Prepare target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["referencia"])

    # Log class distribution
    class_counts = pd.Series(y).value_counts()
    logger.info(f"Number of classes: {len(class_counts)}")
    logger.info(
        f"Class distribution - min: {class_counts.min()}, max: {class_counts.max()}, mean: {class_counts.mean():.2f}")

    preprocessors = {
        'cat_preprocessor': cat_preprocessor,
        'text_preprocessor': text_preprocessor
    }

    return X_all, y, preprocessors, label_encoder


def main():
    try:
        # Load data
        logger.info(f"Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded {len(df)} rows")

        # Basic column checks
        required_cols = ["maker", "series",
                         "model", "descripcion", "referencia"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Prepare features and target with very limited data
        X_all, y, preprocessors, label_encoder = prepare_data(
            df, min_class_count=10, max_classes=50, sample_size=2000)

        # Train/test split
        logger.info("Preparing train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y,
            test_size=0.2,
            random_state=42
        )

        # Use a simple model - Decision Tree
        logger.info("Training a simple Decision Tree model...")
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

        # Train the model
        logger.info("Starting model training...")
        clf.fit(X_train, y_train)
        logger.info("Model training completed successfully")

        # Evaluate
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        logger.info(f"Train accuracy: {train_acc:.3f}")
        logger.info(f"Test accuracy: {test_acc:.3f}")

        # Save artifacts
        logger.info("Saving model artifacts...")
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(preprocessors, PREPROCESSOR_PATH)
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
        logger.info(
            f"Artifacts saved: {MODEL_PATH}, {PREPROCESSOR_PATH}, {LABEL_ENCODER_PATH}")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
