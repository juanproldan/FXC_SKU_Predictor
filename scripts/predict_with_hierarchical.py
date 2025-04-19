import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
import string
import re
from typing import Dict, List, Tuple, Union

# --- Logging setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Paths ---
MODEL_DIR = "models/hierarchical"
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.joblib")

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

class HierarchicalSKUPredictor:
    """A hierarchical model that first predicts maker, then SKU within that maker."""
    
    def __init__(self, model_dir=MODEL_DIR):
        """Initialize the predictor by loading all models and preprocessors."""
        logger.info(f"Initializing HierarchicalSKUPredictor from {model_dir}")
        
        # Load metadata
        self.metadata = joblib.load(METADATA_PATH)
        logger.info(f"Loaded metadata with {len(self.metadata['maker_names'])} makers")
        
        # Load maker model and preprocessors
        self.maker_model = joblib.load(self.metadata['maker_model_path'])
        self.maker_preprocessors = joblib.load(self.metadata['maker_preprocessors_path'])
        self.maker_encoder = self.maker_preprocessors['label_encoder']
        
        # Load SKU models for each maker
        self.sku_models = {}
        self.sku_preprocessors = {}
        self.sku_encoders = {}
        
        for maker in self.metadata['maker_names']:
            maker_dir = os.path.join(model_dir, maker.lower().replace(' ', '_'))
            model_path = os.path.join(maker_dir, "sku_model.joblib")
            preprocessors_path = os.path.join(maker_dir, "sku_preprocessors.joblib")
            encoder_path = os.path.join(maker_dir, "sku_encoder.joblib")
            
            # Only load if the model exists
            if os.path.exists(model_path):
                self.sku_models[maker] = joblib.load(model_path)
                self.sku_preprocessors[maker] = joblib.load(preprocessors_path)
                self.sku_encoders[maker] = joblib.load(encoder_path)
                logger.info(f"Loaded SKU model for {maker}")
            else:
                logger.warning(f"No SKU model found for {maker}")
        
        logger.info(f"Loaded {len(self.sku_models)} SKU models")
    
    def preprocess_input(self, input_data: Dict) -> Dict:
        """Preprocess the input data."""
        processed = {}
        
        # Ensure all required fields are present
        required_fields = ["maker", "series", "model", "descripcion"]
        for field in required_fields:
            if field in input_data:
                processed[field] = preprocess_text(input_data[field])
            else:
                processed[field] = ""
        
        return processed
    
    def predict_maker(self, input_data: Dict) -> Tuple[str, float]:
        """Predict the maker for the input data."""
        # Preprocess input
        processed = self.preprocess_input(input_data)
        
        # Extract text features
        text_preprocessor = self.maker_preprocessors['text_preprocessor']
        X_text = text_preprocessor.transform([processed["descripcion"]]).toarray()
        
        # Predict maker
        maker_probs = self.maker_model.predict_proba(X_text)[0]
        maker_idx = maker_probs.argmax()
        maker_confidence = maker_probs[maker_idx]
        maker_name = self.maker_encoder.inverse_transform([maker_idx])[0]
        
        return maker_name, maker_confidence
    
    def predict_sku(self, maker: str, input_data: Dict) -> Tuple[str, float]:
        """Predict the SKU for a given maker."""
        # Check if we have a model for this maker
        if maker not in self.sku_models:
            logger.warning(f"No SKU model available for maker: {maker}")
            return None, 0.0
        
        # Preprocess input
        processed = self.preprocess_input(input_data)
        
        # Get preprocessors for this maker
        preprocessors = self.sku_preprocessors[maker]
        cat_preprocessor = preprocessors['cat_preprocessor']
        text_preprocessor = preprocessors['text_preprocessor']
        
        # Extract features
        X_cat = cat_preprocessor.transform([[processed["series"], processed["model"]]])
        X_text = text_preprocessor.transform([processed["descripcion"]]).toarray()
        X = np.hstack([X_cat, X_text])
        
        # Predict SKU
        sku_probs = self.sku_models[maker].predict_proba(X)[0]
        sku_idx = sku_probs.argmax()
        sku_confidence = sku_probs[sku_idx]
        sku = self.sku_encoders[maker].inverse_transform([sku_idx])[0]
        
        return sku, sku_confidence
    
    def predict(self, input_data: Dict) -> Dict:
        """Make a full prediction using the hierarchical model."""
        # If maker is provided and we have a model for it, use it directly
        if "maker" in input_data and input_data["maker"] and input_data["maker"] in self.sku_models:
            maker = input_data["maker"]
            maker_confidence = 1.0  # We're using the provided maker
        else:
            # Predict maker
            maker, maker_confidence = self.predict_maker(input_data)
        
        # Predict SKU for the maker
        sku, sku_confidence = self.predict_sku(maker, input_data)
        
        # Calculate overall confidence
        overall_confidence = maker_confidence * sku_confidence if sku else 0.0
        
        return {
            "predicted_maker": maker,
            "maker_confidence": maker_confidence,
            "predicted_sku": sku,
            "sku_confidence": sku_confidence,
            "overall_confidence": overall_confidence
        }

def predict_from_csv(predictor, input_file, output_file=None):
    """Make predictions for all rows in a CSV file."""
    # Load input data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} rows from {input_file}")
    
    # Make predictions
    results = []
    for i, row in df.iterrows():
        input_data = row.to_dict()
        prediction = predictor.predict(input_data)
        
        # Add original data to results
        result = {**input_data, **prediction}
        results.append(result)
        
        # Log progress
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(df)} rows")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results if output file is provided
    if output_file:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    
    return results_df

def predict_single_item(predictor, input_data):
    """Make a prediction for a single item."""
    prediction = predictor.predict(input_data)
    
    # Print prediction
    print("\n--- Prediction Results ---")
    print(f"Input: {input_data}")
    print(f"Predicted Maker: {prediction['predicted_maker']} (confidence: {prediction['maker_confidence']:.3f})")
    print(f"Predicted SKU: {prediction['predicted_sku']} (confidence: {prediction['sku_confidence']:.3f})")
    print(f"Overall Confidence: {prediction['overall_confidence']:.3f}")
    
    return prediction

def main():
    try:
        # Check if models exist
        if not os.path.exists(METADATA_PATH):
            logger.error(f"Model metadata not found at {METADATA_PATH}. Please train the model first.")
            return
        
        # Initialize predictor
        predictor = HierarchicalSKUPredictor()
        
        # Parse command line arguments
        if len(sys.argv) > 1:
            # CSV file input
            if sys.argv[1].endswith('.csv'):
                input_file = sys.argv[1]
                output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.csv', '_predictions.csv')
                predict_from_csv(predictor, input_file, output_file)
            else:
                # Interactive mode with command line arguments
                input_data = {
                    "maker": sys.argv[1] if len(sys.argv) > 1 else "",
                    "series": sys.argv[2] if len(sys.argv) > 2 else "",
                    "model": sys.argv[3] if len(sys.argv) > 3 else "",
                    "descripcion": sys.argv[4] if len(sys.argv) > 4 else ""
                }
                predict_single_item(predictor, input_data)
        else:
            # Interactive mode
            print("Enter item details for prediction:")
            input_data = {
                "maker": input("Maker (optional): "),
                "series": input("Series (optional): "),
                "model": input("Model (optional): "),
                "descripcion": input("Description (required): ")
            }
            predict_single_item(predictor, input_data)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
