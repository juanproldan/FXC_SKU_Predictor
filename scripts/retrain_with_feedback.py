import os
import sys
import joblib
import pandas as pd
import numpy as np
import string
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feedback_db import get_unprocessed_feedback, mark_feedback_as_processed, save_retraining_record

# --- Constants ---
DATA_PATH = "Fuente_Json_Consolidado/clean_training_data.csv"
MIN_EXAMPLES_PER_SKU = 10
MAX_SKUS = 50
MODEL_DIR = "models/renault_neural"
os.makedirs(MODEL_DIR, exist_ok=True)

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

def load_feedback_data():
    """Load feedback data from the database."""
    print("Loading feedback data...")
    feedback_items = get_unprocessed_feedback()
    
    if not feedback_items:
        print("No unprocessed feedback found.")
        return None, []
    
    print(f"Found {len(feedback_items)} unprocessed feedback items.")
    
    # Convert to DataFrame
    feedback_df = pd.DataFrame(feedback_items)
    
    # Filter to only include Renault feedback
    feedback_df = feedback_df[feedback_df["maker"].str.lower() == "renault"].reset_index(drop=True)
    
    if len(feedback_df) == 0:
        print("No Renault feedback found.")
        return None, []
    
    print(f"Found {len(feedback_df)} Renault feedback items.")
    
    # Create training data from feedback
    training_data = []
    
    # Process correct predictions
    correct_feedback = feedback_df[feedback_df["is_correct"] == 1]
    for _, row in correct_feedback.iterrows():
        training_data.append({
            "maker": row["maker"],
            "series": row["series"],
            "model": row["model_year"],
            "descripcion": row["description"],
            "referencia": row["predicted_sku"]
        })
    
    # Process incorrect predictions with corrections
    incorrect_feedback = feedback_df[(feedback_df["is_correct"] == 0) & (feedback_df["correct_sku"].notnull())]
    for _, row in incorrect_feedback.iterrows():
        training_data.append({
            "maker": row["maker"],
            "series": row["series"],
            "model": row["model_year"],
            "descripcion": row["description"],
            "referencia": row["correct_sku"]
        })
    
    # Convert to DataFrame
    if training_data:
        feedback_training_df = pd.DataFrame(training_data)
        print(f"Created {len(feedback_training_df)} training examples from feedback.")
        return feedback_training_df, [item["id"] for item in feedback_items]
    else:
        print("No usable training data in feedback.")
        return None, []

def main():
    print(f"Loading original training data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    
    # Preprocess text columns
    for col in ["maker", "series", "model", "descripcion"]:
        df[col] = df[col].fillna('').astype(str).apply(preprocess_text)
    
    # Filter to only include Renault (case-insensitive)
    df_renault = df[df["maker"].str.lower() == "renault"].reset_index(drop=True)
    print(f"Filtered to {len(df_renault)} Renault rows")
    
    # Load feedback data
    feedback_df, feedback_ids = load_feedback_data()
    
    if feedback_df is not None and len(feedback_df) > 0:
        # Preprocess feedback text
        for col in ["maker", "series", "model", "descripcion"]:
            if col in feedback_df.columns:
                feedback_df[col] = feedback_df[col].fillna('').astype(str).apply(preprocess_text)
        
        # Combine original data with feedback data
        df_combined = pd.concat([df_renault, feedback_df], ignore_index=True)
        print(f"Combined dataset size: {len(df_combined)} rows")
    else:
        df_combined = df_renault
        print("Using original dataset only.")
    
    # Filter SKUs with too few examples
    sku_counts = df_combined["referencia"].value_counts()
    valid_skus = sku_counts[sku_counts >= MIN_EXAMPLES_PER_SKU].index
    print(f"Keeping {len(valid_skus)} out of {len(sku_counts)} SKUs with at least {MIN_EXAMPLES_PER_SKU} examples")
    
    # Limit to top SKUs
    if len(valid_skus) > MAX_SKUS:
        valid_skus = sku_counts.nlargest(MAX_SKUS).index
        print(f"Limiting to top {MAX_SKUS} SKUs")
    
    df_filtered = df_combined[df_combined["referencia"].isin(valid_skus)].reset_index(drop=True)
    print(f"Dataset size after filtering: {len(df_filtered)} rows")
    
    # Encode SKUs
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    
    # Check if we need to update the label encoder
    new_skus = set(df_filtered["referencia"].unique()) - set(label_encoder.classes_)
    if new_skus:
        print(f"Found {len(new_skus)} new SKUs in feedback data. Updating label encoder.")
        # Create a new label encoder with all SKUs
        from sklearn.preprocessing import LabelEncoder
        new_label_encoder = LabelEncoder()
        new_label_encoder.fit(df_filtered["referencia"])
        label_encoder = new_label_encoder
    
    y = label_encoder.transform(df_filtered["referencia"])
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    
    # Extract features using TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df_filtered["descripcion"])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load the current model to get its accuracy
    try:
        current_model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
        current_vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
        
        # Transform test data with current vectorizer
        X_test_current = current_vectorizer.transform(df_filtered.loc[X_test.index, "descripcion"])
        
        # Evaluate current model
        current_accuracy = current_model.score(X_test_current, y_test)
        print(f"Current model accuracy: {current_accuracy:.3f}")
    except Exception as e:
        print(f"Error evaluating current model: {str(e)}")
        current_accuracy = 0.0
    
    # Train a neural network model (MLP)
    print("Training neural network model...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True,
        random_state=42
    )
    
    mlp.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = mlp.predict(X_test)
    new_accuracy = accuracy_score(y_test, y_pred)
    print(f"New model accuracy: {new_accuracy:.3f}")
    
    # Only save the model if it's better than the current one
    if new_accuracy > current_accuracy:
        print(f"New model is better ({new_accuracy:.3f} > {current_accuracy:.3f}). Saving...")
        
        # Save model and vectorizer
        model_path = os.path.join(MODEL_DIR, "model.joblib")
        vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
        encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
        metadata_path = os.path.join(MODEL_DIR, "metadata.joblib")
        
        print(f"Saving model to {model_path}")
        joblib.dump(mlp, model_path)
        
        print(f"Saving vectorizer to {vectorizer_path}")
        joblib.dump(vectorizer, vectorizer_path)
        
        print(f"Saving label encoder to {encoder_path}")
        joblib.dump(label_encoder, encoder_path)
        
        # Save metadata
        metadata = {
            'min_examples_per_sku': MIN_EXAMPLES_PER_SKU,
            'max_skus': MAX_SKUS,
            'num_skus': len(valid_skus),
            'skus': valid_skus.tolist(),
            'test_accuracy': new_accuracy,
            'last_retrained': datetime.now().isoformat()
        }
        
        print(f"Saving metadata to {metadata_path}")
        joblib.dump(metadata, metadata_path)
        
        # Save retraining record
        record = {
            'timestamp': datetime.now().isoformat(),
            'feedback_count': len(feedback_df) if feedback_df is not None else 0,
            'previous_accuracy': current_accuracy,
            'new_accuracy': new_accuracy,
            'model_path': model_path
        }
        save_retraining_record(record)
        
        print("Model saved successfully.")
    else:
        print(f"New model is not better ({new_accuracy:.3f} <= {current_accuracy:.3f}). Keeping current model.")
    
    # Mark feedback as processed
    if feedback_ids:
        mark_feedback_as_processed(feedback_ids)
        print(f"Marked {len(feedback_ids)} feedback items as processed.")

if __name__ == "__main__":
    main()
