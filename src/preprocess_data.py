# src/preprocess_data.py

import pandas as pd
import os
import logging
import argparse
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # Use joblib for saving sklearn objects

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(input_csv_path: str, output_dir: str, test_size: float = 0.15, val_size: float = 0.15):
    """
    Loads raw correlated data, cleans text, preprocesses features (TF-IDF for text, OHE for categoricals),
    encodes labels, splits data, fits preprocessor, and saves processed data and artifacts.
    """
    if not os.path.exists(input_csv_path):
        logging.error(f"Input file not found: {input_csv_path}")
        return

    logging.info(f"Loading raw correlated data from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
        logging.info(f"Loaded {len(df)} rows.")
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        return

    # --- Feature and Target Selection --- 
    # We will clean 'description' to create 'description_cleaned'
    features = ['description_cleaned', 'maker', 'model'] 
    target = 'SKU'

    # Basic Cleaning & Type Conversion
    logging.info("Performing basic cleaning and type conversion...")
    # Ensure target exists before dropping rows
    df = df.dropna(subset=[target])
    df['model'] = df['model'].fillna('missing').astype(str) # Fill NaN and convert model year to string
    df['maker'] = df['maker'].fillna('missing').astype(str) # Fill NaN maker
    df['description'] = df['description'].fillna('missing').astype(str) # Fill NaN original description

    # --- Text Cleaning for 'description' column --- 
    logging.info("Cleaning 'description' column...")
    def clean_text(text):
        text = text.lower() # Lowercase
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation/special chars except whitespace
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        return text
    
    df['description_cleaned'] = df['description'].apply(clean_text)
    
    # Now drop rows if any of the final features are missing AFTER cleaning/filling
    df = df.dropna(subset=features) 
    
    logging.info(f"Data shape after initial cleaning, text cleaning, and NA handling: {df.shape}")

    # --- Filter out single-occurrence SKUs for stratification --- 
    logging.info("Filtering out SKUs that appear only once...")
    sku_counts = df[target].value_counts()
    skus_to_keep = sku_counts[sku_counts > 1].index
    original_rows = len(df)
    df_filtered = df[df[target].isin(skus_to_keep)]
    removed_rows = original_rows - len(df_filtered)
    removed_skus = len(sku_counts) - len(skus_to_keep)
    logging.info(f"Removed {removed_rows} rows corresponding to {removed_skus} single-occurrence SKUs.")
    logging.info(f"Data shape after filtering single-occurrence SKUs: {df_filtered.shape}")

    if df_filtered.empty:
        logging.error("No data remaining after filtering single-occurrence SKUs. Cannot proceed.")
        return

    X = df_filtered[features]
    y = df_filtered[target]

    # --- Data Splitting ---
    # Split into Train+Val and Test
    # Stratify based on the filtered y
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    # Split Train+Val into Train and Validation
    # Adjust val_size relative to the size of X_train_val
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=42, stratify=y_train_val
    )
    logging.info(f"Data split complete:")
    logging.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    # --- Label Encoding ---
    logging.info("Encoding target variable (SKU)...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    logging.info(f"Number of unique SKU classes: {len(label_encoder.classes_)}")

    # --- Feature Preprocessing Pipeline ---
    logging.info("Setting up feature preprocessing pipeline...")
    # Define transformers for different column types
    text_features = ['description_cleaned']
    categorical_features = ['maker', 'model']

    # Create preprocessing steps
    text_transformer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000) # Limit features for memory
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

    # Use ColumnTransformer to apply different transformers to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', text_transformer, text_features[0]), # Pass column name string directly
            ('onehot', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop columns not specified
    )

    # --- Fit and Transform ---
    logging.info("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    logging.info("Transforming validation and test data...")
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    logging.info(f"Processed feature shapes: Train={X_train_processed.shape}, Val={X_val_processed.shape}, Test={X_test_processed.shape}")

    # --- Save Artifacts and Processed Data ---
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving preprocessor, label encoder, and processed data to {output_dir}")

    try:
        # Save the fitted preprocessor pipeline
        joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
        # Save the fitted label encoder
        joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
        # Save processed data (sparse matrix + labels)
        joblib.dump({'X': X_train_processed, 'y': y_train_encoded}, os.path.join(output_dir, 'train_processed.joblib'))
        joblib.dump({'X': X_val_processed, 'y': y_val_encoded}, os.path.join(output_dir, 'val_processed.joblib'))
        joblib.dump({'X': X_test_processed, 'y': y_test_encoded}, os.path.join(output_dir, 'test_processed.joblib'))

        logging.info("Preprocessing complete. Artifacts and data saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save artifacts or data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for SKU prediction.")
    parser.add_argument(
        "--input",
        default="data/correlated_sku_data.csv", # Changed default input
        help="Path to the input raw correlated CSV file."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save the preprocessor, label encoder, and processed data splits."
    )
    parser.add_argument(
        "--test-size",
        type=float, default=0.15,
        help="Proportion of the dataset to include in the test split."
    )
    parser.add_argument(
        "--val-size",
        type=float, default=0.15,
        help="Proportion of the dataset to include in the validation split."
    )

    args = parser.parse_args()

    # Determine project root assuming script is in 'src' subdir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Construct absolute paths relative to project root if needed
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(project_root, input_path)

    output_path = args.output_dir
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)

    preprocess_data(input_path, output_path, args.test_size, args.val_size)
