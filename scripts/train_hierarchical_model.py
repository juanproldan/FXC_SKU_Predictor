import os
import sys
import traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from typing import Tuple, Dict, List
import logging
import re
import string

# --- Logging setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Paths ---
DATA_PATH = "Fuente_Json_Consolidado/item_training_data.csv"
MODEL_DIR = "models/hierarchical"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Constants ---
MIN_EXAMPLES_PER_SKU = 3  # Minimum examples required for an SKU to be included
MIN_EXAMPLES_PER_MAKER = 100  # Minimum examples required for a maker to have its own model

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

def prepare_maker_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Prepare features and target for maker prediction."""
    logger.info("Preparing data for maker prediction...")
    
    # Preprocess text columns
    for col in ["maker", "series", "model", "descripcion"]:
        df[col] = df[col].fillna('').astype(str).apply(preprocess_text)
    
    # Text features for maker prediction
    text_preprocessor = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        strip_accents='unicode',
        analyzer='word'
    )
    X_text = text_preprocessor.fit_transform(df["descripcion"]).toarray()
    logger.info(f"Text features shape: {X_text.shape}")
    
    # Prepare target for maker prediction
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["maker"])
    
    # Log maker distribution
    maker_counts = pd.Series(df["maker"]).value_counts()
    logger.info(f"Number of makers: {len(maker_counts)}")
    logger.info(f"Maker distribution: {maker_counts.to_dict()}")
    
    preprocessors = {
        'text_preprocessor': text_preprocessor,
        'label_encoder': label_encoder
    }
    
    return X_text, y, preprocessors

def prepare_sku_data(df: pd.DataFrame, min_examples: int = 3) -> Tuple[np.ndarray, np.ndarray, Dict, LabelEncoder]:
    """Prepare features and target for SKU prediction within a maker."""
    logger.info(f"Preparing SKU data with minimum {min_examples} examples per SKU...")
    
    # Preprocess text columns if not already done
    for col in ["maker", "series", "model", "descripcion"]:
        if not df[col].dtype == object or not all(isinstance(x, str) for x in df[col].sample(min(10, len(df)))):
            df[col] = df[col].fillna('').astype(str).apply(preprocess_text)
    
    # Filter SKUs with too few examples
    if min_examples > 1:
        logger.info(f"Filtering SKUs with fewer than {min_examples} examples")
        sku_counts = df["referencia"].value_counts()
        valid_skus = sku_counts[sku_counts >= min_examples].index
        logger.info(f"Keeping {len(valid_skus)} out of {len(sku_counts)} SKUs")
        df = df[df["referencia"].isin(valid_skus)].reset_index(drop=True)
        logger.info(f"Dataset size after filtering: {len(df)} rows")
    
    # Categorical features
    cat_preprocessor = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False,
        min_frequency=2
    )
    X_cat = cat_preprocessor.fit_transform(df[["series", "model"]])
    logger.info(f"Categorical features shape: {X_cat.shape}")
    
    # Text features
    text_preprocessor = TfidfVectorizer(
        max_features=800,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
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
    logger.info(f"Number of SKU classes: {len(class_counts)}")
    logger.info(
        f"SKU class distribution - min: {class_counts.min()}, max: {class_counts.max()}, mean: {class_counts.mean():.2f}")
    
    preprocessors = {
        'cat_preprocessor': cat_preprocessor,
        'text_preprocessor': text_preprocessor
    }
    
    return X_all, y, preprocessors, label_encoder

def train_maker_model(X, y, preprocessors):
    """Train a model to predict the maker."""
    logger.info("Training maker prediction model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Fitting maker model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logger.info(f"Maker model - Train accuracy: {train_acc:.3f}")
    logger.info(f"Maker model - Test accuracy: {test_acc:.3f}")
    
    # Save detailed classification report
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save model and preprocessors
    maker_model_path = os.path.join(MODEL_DIR, "maker_model.joblib")
    maker_preprocessors_path = os.path.join(MODEL_DIR, "maker_preprocessors.joblib")
    
    joblib.dump(model, maker_model_path)
    joblib.dump(preprocessors, maker_preprocessors_path)
    
    logger.info(f"Maker model saved to {maker_model_path}")
    
    return model, preprocessors, report

def train_sku_model_for_maker(maker_name, maker_df, min_examples=MIN_EXAMPLES_PER_SKU):
    """Train a SKU prediction model for a specific maker."""
    logger.info(f"Training SKU model for maker: {maker_name}")
    logger.info(f"Data size for {maker_name}: {len(maker_df)} rows")
    
    # Check if we have enough data
    if len(maker_df) < MIN_EXAMPLES_PER_MAKER:
        logger.warning(f"Not enough data for {maker_name}. Skipping.")
        return None, None, None, None
    
    # Prepare data
    X, y, preprocessors, label_encoder = prepare_sku_data(maker_df, min_examples)
    
    # Check if we have enough SKUs after filtering
    if len(np.unique(y)) < 2:
        logger.warning(f"Not enough unique SKUs for {maker_name} after filtering. Skipping.")
        return None, None, None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    logger.info(f"Fitting SKU model for {maker_name}...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logger.info(f"{maker_name} SKU model - Train accuracy: {train_acc:.3f}")
    logger.info(f"{maker_name} SKU model - Test accuracy: {test_acc:.3f}")
    
    # Save model and preprocessors
    maker_dir = os.path.join(MODEL_DIR, maker_name.lower().replace(' ', '_'))
    os.makedirs(maker_dir, exist_ok=True)
    
    model_path = os.path.join(maker_dir, "sku_model.joblib")
    preprocessors_path = os.path.join(maker_dir, "sku_preprocessors.joblib")
    encoder_path = os.path.join(maker_dir, "sku_encoder.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessors, preprocessors_path)
    joblib.dump(label_encoder, encoder_path)
    
    logger.info(f"{maker_name} SKU model saved to {model_path}")
    
    return model, preprocessors, label_encoder, test_acc

def main():
    try:
        # Load data
        logger.info(f"Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded {len(df)} rows")
        
        # Basic column checks
        required_cols = ["maker", "series", "model", "descripcion", "referencia"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Step 1: Train maker prediction model
        X_maker, y_maker, maker_preprocessors = prepare_maker_data(df)
        maker_model, _, maker_report = train_maker_model(X_maker, y_maker, maker_preprocessors)
        
        # Get maker names
        maker_encoder = maker_preprocessors['label_encoder']
        maker_names = maker_encoder.classes_
        
        # Step 2: Train SKU models for each maker
        maker_accuracies = {}
        for maker_idx, maker_name in enumerate(maker_names):
            logger.info(f"Processing maker {maker_idx+1}/{len(maker_names)}: {maker_name}")
            
            # Filter data for this maker
            maker_df = df[df["maker"] == maker_name].reset_index(drop=True)
            
            # Train SKU model for this maker
            _, _, _, test_acc = train_sku_model_for_maker(maker_name, maker_df)
            
            if test_acc is not None:
                maker_accuracies[maker_name] = test_acc
        
        # Log overall results
        logger.info("Training completed successfully")
        logger.info("Maker model accuracy: {:.3f}".format(maker_report['accuracy']))
        logger.info("SKU model accuracies by maker:")
        for maker, acc in maker_accuracies.items():
            logger.info(f"  {maker}: {acc:.3f}")
        
        # Save metadata about the hierarchical model
        metadata = {
            'maker_model_path': os.path.join(MODEL_DIR, "maker_model.joblib"),
            'maker_preprocessors_path': os.path.join(MODEL_DIR, "maker_preprocessors.joblib"),
            'maker_names': maker_names.tolist(),
            'maker_accuracies': maker_accuracies,
            'min_examples_per_sku': MIN_EXAMPLES_PER_SKU
        }
        
        metadata_path = os.path.join(MODEL_DIR, "metadata.joblib")
        joblib.dump(metadata, metadata_path)
        logger.info(f"Model metadata saved to {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
