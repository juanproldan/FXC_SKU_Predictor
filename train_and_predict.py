import pandas as pd
import numpy as np
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Constants ---
DATA_PATH = "Fuente_Json_Consolidado/item_training_data.csv"
MIN_EXAMPLES_PER_SKU = 5
MAX_SKUS_PER_MAKER = 100

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
    
    # Filter to only include Renault (the most common maker)
    df_renault = df[df["maker"] == "renault"].reset_index(drop=True)
    print(f"Filtered to {len(df_renault)} Renault rows")
    
    # Filter SKUs with too few examples
    sku_counts = df_renault["referencia"].value_counts()
    valid_skus = sku_counts[sku_counts >= MIN_EXAMPLES_PER_SKU].index
    print(f"Keeping {len(valid_skus)} out of {len(sku_counts)} SKUs")
    
    # Limit to top SKUs
    if len(valid_skus) > MAX_SKUS_PER_MAKER:
        valid_skus = sku_counts.nlargest(MAX_SKUS_PER_MAKER).index
        print(f"Limiting to top {MAX_SKUS_PER_MAKER} SKUs")
    
    df_filtered = df_renault[df_renault["referencia"].isin(valid_skus)].reset_index(drop=True)
    print(f"Dataset size after filtering: {len(df_filtered)} rows")
    
    # Extract features
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df_filtered["descripcion"]).toarray()
    y = df_filtered["referencia"].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    
    # Make predictions for sample descriptions
    sample_descriptions = [
        "amortiguador delantero renault logan 2015",
        "filtro de aceite renault clio 2010",
        "pastillas de freno renault sandero 2018",
        "bujia renault duster 2016"
    ]
    
    print("\nMaking predictions for sample descriptions:")
    for desc in sample_descriptions:
        # Preprocess
        processed_desc = preprocess_text(desc)
        
        # Transform
        X_desc = vectorizer.transform([processed_desc]).toarray()
        
        # Predict
        probs = model.predict_proba(X_desc)[0]
        top_indices = probs.argsort()[-3:][::-1]
        
        print(f"\nDescription: {desc}")
        print("Top 3 predictions:")
        for idx in top_indices:
            sku = model.classes_[idx]
            confidence = probs[idx]
            
            # Find a sample row with this SKU to show more info
            sample_row = df_filtered[df_filtered["referencia"] == sku].iloc[0]
            maker = sample_row["maker"]
            series = sample_row["series"]
            model_year = sample_row["model"]
            
            print(f"  SKU: {sku} ({maker} {series} {model_year}) - confidence: {confidence:.3f}")

if __name__ == "__main__":
    main()
