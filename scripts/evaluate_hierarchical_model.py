import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from predict_with_hierarchical import HierarchicalSKUPredictor

# --- Logging setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Paths ---
DATA_PATH = "Fuente_Json_Consolidado/item_training_data.csv"
MODEL_DIR = "models/hierarchical"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_model(test_data, predictor):
    """Evaluate the hierarchical model on test data."""
    logger.info(f"Evaluating model on {len(test_data)} test samples")
    
    # Make predictions
    maker_predictions = []
    sku_predictions = []
    maker_true = []
    sku_true = []
    
    for i, row in test_data.iterrows():
        input_data = row.to_dict()
        prediction = predictor.predict(input_data)
        
        maker_predictions.append(prediction['predicted_maker'])
        maker_true.append(row['maker'])
        
        if prediction['predicted_sku'] is not None:
            sku_predictions.append(prediction['predicted_sku'])
            sku_true.append(row['referencia'])
        
        # Log progress
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(test_data)} test samples")
    
    # Calculate maker accuracy
    maker_accuracy = accuracy_score(maker_true, maker_predictions)
    logger.info(f"Maker prediction accuracy: {maker_accuracy:.3f}")
    
    # Calculate SKU accuracy (only for samples where we made a prediction)
    if sku_predictions:
        sku_accuracy = accuracy_score(sku_true, sku_predictions)
        logger.info(f"SKU prediction accuracy: {sku_accuracy:.3f}")
        logger.info(f"SKU prediction coverage: {len(sku_predictions)/len(test_data):.3f}")
    else:
        sku_accuracy = 0.0
        logger.warning("No SKU predictions were made")
    
    # Generate detailed reports
    maker_report = classification_report(maker_true, maker_predictions, output_dict=True)
    
    # Save results
    results = {
        'maker_accuracy': maker_accuracy,
        'sku_accuracy': sku_accuracy,
        'sku_coverage': len(sku_predictions)/len(test_data) if sku_predictions else 0,
        'maker_report': maker_report,
        'test_size': len(test_data)
    }
    
    return results

def analyze_errors(test_data, predictor):
    """Analyze prediction errors to identify patterns."""
    logger.info("Analyzing prediction errors...")
    
    # Make predictions and collect errors
    maker_errors = []
    sku_errors = []
    
    for i, row in test_data.iterrows():
        input_data = row.to_dict()
        prediction = predictor.predict(input_data)
        
        # Check maker prediction
        if prediction['predicted_maker'] != row['maker']:
            maker_errors.append({
                'true_maker': row['maker'],
                'predicted_maker': prediction['predicted_maker'],
                'confidence': prediction['maker_confidence'],
                'description': row['descripcion']
            })
        
        # Check SKU prediction (only if maker is correct)
        elif prediction['predicted_sku'] is not None and prediction['predicted_sku'] != row['referencia']:
            sku_errors.append({
                'maker': row['maker'],
                'true_sku': row['referencia'],
                'predicted_sku': prediction['predicted_sku'],
                'confidence': prediction['sku_confidence'],
                'description': row['descripcion']
            })
        
        # Log progress
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(test_data)} samples for error analysis")
    
    # Analyze maker errors
    maker_error_df = pd.DataFrame(maker_errors)
    if not maker_error_df.empty:
        logger.info(f"Found {len(maker_error_df)} maker prediction errors")
        
        # Most common misclassifications
        confusion = maker_error_df.groupby(['true_maker', 'predicted_maker']).size().reset_index(name='count')
        confusion = confusion.sort_values('count', ascending=False)
        logger.info("Top maker misclassifications:")
        for _, row in confusion.head(5).iterrows():
            logger.info(f"  {row['true_maker']} -> {row['predicted_maker']}: {row['count']} times")
        
        # Save maker errors
        maker_error_path = os.path.join(RESULTS_DIR, "maker_errors.csv")
        maker_error_df.to_csv(maker_error_path, index=False)
        logger.info(f"Maker errors saved to {maker_error_path}")
    
    # Analyze SKU errors
    sku_error_df = pd.DataFrame(sku_errors)
    if not sku_error_df.empty:
        logger.info(f"Found {len(sku_error_df)} SKU prediction errors")
        
        # Group by maker
        maker_sku_errors = sku_error_df.groupby('maker').size().reset_index(name='error_count')
        maker_sku_errors = maker_sku_errors.sort_values('error_count', ascending=False)
        logger.info("SKU errors by maker:")
        for _, row in maker_sku_errors.iterrows():
            logger.info(f"  {row['maker']}: {row['error_count']} errors")
        
        # Save SKU errors
        sku_error_path = os.path.join(RESULTS_DIR, "sku_errors.csv")
        sku_error_df.to_csv(sku_error_path, index=False)
        logger.info(f"SKU errors saved to {sku_error_path}")
    
    return {
        'maker_errors': maker_error_df if not maker_error_df.empty else None,
        'sku_errors': sku_error_df if not sku_error_df.empty else None
    }

def main():
    try:
        # Check if model exists
        metadata_path = os.path.join(MODEL_DIR, "metadata.joblib")
        if not os.path.exists(metadata_path):
            logger.error(f"Model metadata not found at {metadata_path}. Please train the model first.")
            return
        
        # Load test data
        logger.info(f"Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded {len(df)} rows")
        
        # Create a test set (20% of data)
        test_size = min(1000, int(len(df) * 0.2))  # Limit to 1000 samples for faster evaluation
        test_data = df.sample(test_size, random_state=42)
        logger.info(f"Created test set with {len(test_data)} samples")
        
        # Initialize predictor
        predictor = HierarchicalSKUPredictor()
        
        # Evaluate model
        results = evaluate_model(test_data, predictor)
        
        # Analyze errors
        error_analysis = analyze_errors(test_data, predictor)
        
        # Save overall results
        results_path = os.path.join(RESULTS_DIR, "evaluation_results.txt")
        with open(results_path, 'w') as f:
            f.write("Hierarchical SKU Predictor Evaluation\n")
            f.write("====================================\n\n")
            f.write(f"Test set size: {results['test_size']} samples\n\n")
            f.write(f"Maker prediction accuracy: {results['maker_accuracy']:.3f}\n")
            f.write(f"SKU prediction accuracy: {results['sku_accuracy']:.3f}\n")
            f.write(f"SKU prediction coverage: {results['sku_coverage']:.3f}\n\n")
            
            f.write("Maker Classification Report:\n")
            for maker, metrics in results['maker_report'].items():
                if maker not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"  {maker}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1-score={metrics['f1-score']:.3f}, support={metrics['support']}\n")
            
            f.write("\nError Analysis:\n")
            if error_analysis['maker_errors'] is not None:
                f.write(f"  Maker prediction errors: {len(error_analysis['maker_errors'])}\n")
            if error_analysis['sku_errors'] is not None:
                f.write(f"  SKU prediction errors: {len(error_analysis['sku_errors'])}\n")
        
        logger.info(f"Evaluation results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
