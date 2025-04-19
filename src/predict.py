import argparse
import pandas as pd
import joblib
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_preprocessor(preprocessor_path):
    if not os.path.exists(preprocessor_path):
        logging.error(f"Preprocessor not found at: {preprocessor_path}")
        raise FileNotFoundError(f"Preprocessor not found at: {preprocessor_path}")
    return joblib.load(preprocessor_path)

def load_model(model_path):
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

def main():
    parser = argparse.ArgumentParser(description="Make predictions with the trained SKU model.")
    parser.add_argument('--input', required=True, help='Path to new data CSV file (must have same structure as training data)')
    parser.add_argument('--output', default='predictions.csv', help='Path to output CSV file for predictions')
    parser.add_argument('--preprocessor', default='data/processed/preprocessor.joblib', help='Path to saved preprocessor joblib')
    parser.add_argument('--model', default='models/lgbm_sku_predictor.joblib', help='Path to trained model joblib')
    args = parser.parse_args()

    # Load new data
    df = pd.read_csv(args.input)
    logging.info(f"Loaded input data: {df.shape}")

    # Load preprocessor and model
    preprocessor = load_preprocessor(args.preprocessor)
    model = load_model(args.model)

    # Preprocess input data
    X = preprocessor.transform(df)
    logging.info(f"Transformed input data: {X.shape}")

    # Make predictions
    predictions = model.predict(X)
    df['predicted_sku'] = predictions

    # Save predictions
    df.to_csv(args.output, index=False)
    logging.info(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()
