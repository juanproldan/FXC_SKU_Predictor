import os
import joblib

# Paths
MODEL_DIR = "models/hierarchical"
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.joblib")

def main():
    print("Checking if model files exist...")
    
    # Check metadata
    if os.path.exists(METADATA_PATH):
        print(f"Metadata file exists at {METADATA_PATH}")
        metadata = joblib.load(METADATA_PATH)
        print(f"Loaded metadata with {len(metadata['maker_names'])} makers")
        print(f"Maker names: {metadata['maker_names']}")
        print(f"Maker accuracies: {metadata['maker_accuracies']}")
    else:
        print(f"Metadata file not found at {METADATA_PATH}")
    
    # Check maker model
    maker_model_path = os.path.join(MODEL_DIR, "maker_model.joblib")
    if os.path.exists(maker_model_path):
        print(f"Maker model exists at {maker_model_path}")
    else:
        print(f"Maker model not found at {maker_model_path}")
    
    # Check SKU models for each maker
    for maker in os.listdir(MODEL_DIR):
        maker_dir = os.path.join(MODEL_DIR, maker)
        if os.path.isdir(maker_dir) and maker != "__pycache__":
            sku_model_path = os.path.join(maker_dir, "sku_model.joblib")
            if os.path.exists(sku_model_path):
                print(f"SKU model for {maker} exists at {sku_model_path}")
            else:
                print(f"SKU model for {maker} not found at {sku_model_path}")

if __name__ == "__main__":
    main()
