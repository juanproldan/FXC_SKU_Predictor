import os
import pandas as pd
import numpy as np
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

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


def main():
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # Preprocess text columns
    for col in ["maker", "series", "model", "descripcion"]:
        df[col] = df[col].fillna('').astype(str).apply(preprocess_text)

    # Filter to only include Renault (case-insensitive)
    df_renault = df[df["maker"].str.lower(
    ) == "renault"].reset_index(drop=True)
    print(f"Filtered to {len(df_renault)} Renault rows")

    # Filter SKUs with too few examples
    sku_counts = df_renault["referencia"].value_counts()
    valid_skus = sku_counts[sku_counts >= MIN_EXAMPLES_PER_SKU].index
    print(
        f"Keeping {len(valid_skus)} out of {len(sku_counts)} SKUs with at least {MIN_EXAMPLES_PER_SKU} examples")

    # Limit to top SKUs
    if len(valid_skus) > MAX_SKUS:
        valid_skus = sku_counts.nlargest(MAX_SKUS).index
        print(f"Limiting to top {MAX_SKUS} SKUs")

    df_filtered = df_renault[df_renault["referencia"].isin(
        valid_skus)].reset_index(drop=True)
    print(f"Dataset size after filtering: {len(df_filtered)} rows")

    # Encode SKUs
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_filtered["referencia"])
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
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.3f}")

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
        'test_accuracy': accuracy
    }

    print(f"Saving metadata to {metadata_path}")
    joblib.dump(metadata, metadata_path)

    # Print some example SKUs
    print("\nExample SKUs in the model:")
    for sku in valid_skus[:10]:
        examples = df_filtered[df_filtered["referencia"]
                               == sku]["descripcion"].values[:2]
        print(f"  {sku}: {examples}")

    # Compare with Random Forest (our previous best model)
    print("\nTraining Random Forest model for comparison...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    rf_accuracy = rf.score(X_test, y_test)
    print(f"Random Forest test accuracy: {rf_accuracy:.3f}")

    # Save the better model
    if rf_accuracy > accuracy:
        print("Random Forest performed better, saving it as the final model")
        joblib.dump(rf, model_path)
        metadata['test_accuracy'] = rf_accuracy
        joblib.dump(metadata, metadata_path)


if __name__ == "__main__":
    main()
