import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load a small subset of the data
print("Loading data...")
df = pd.read_csv("Fuente_Json_Consolidado/item_training_data.csv")
print(f"Loaded {len(df)} rows")

# Filter to only include Renault and SKUs with at least 10 examples
print("Filtering data...")
df = df[df["maker"] == "renault"]
sku_counts = df["referencia"].value_counts()
common_skus = sku_counts[sku_counts >= 10].index
df = df[df["referencia"].isin(common_skus)]
print(f"Filtered to {len(df)} rows with {len(common_skus)} SKUs")

# Take only the top 20 SKUs
top_skus = sku_counts.nlargest(20).index
df = df[df["referencia"].isin(top_skus)]
print(f"Using top 20 SKUs with {len(df)} rows")

# Prepare features and target
print("Preparing features...")
vectorizer = CountVectorizer(max_features=100)
X = vectorizer.fit_transform(df["descripcion"]).toarray()
y = df["referencia"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=10, random_state=42)
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
    X_desc = vectorizer.transform([desc]).toarray()
    pred = model.predict(X_desc)[0]
    probs = model.predict_proba(X_desc)[0]
    top_idx = probs.argmax()
    confidence = probs[top_idx]
    
    print(f"\nDescription: {desc}")
    print(f"Predicted SKU: {pred}")
    print(f"Confidence: {confidence:.3f}")
    
    # Show top 3 predictions
    top_indices = probs.argsort()[-3:][::-1]
    print("Top 3 predictions:")
    for idx in top_indices:
        sku = model.classes_[idx]
        conf = probs[idx]
        print(f"  {sku}: {conf:.3f}")
