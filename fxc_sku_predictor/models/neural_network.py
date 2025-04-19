"""
Neural network model for SKU prediction.

This module provides functions for loading and using the neural network model
for predicting SKUs from product descriptions.
"""

import os
import joblib
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Import text preprocessing
from fxc_sku_predictor.utils.text_preprocessing import preprocess_text

# Set up logging
logger = logging.getLogger(__name__)

# --- Constants ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "models/multi_maker_neural")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.joblib")

# --- Model Loading ---


def load_model() -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Load the neural network model and associated objects.

    Returns:
        Tuple containing:
            - model: The trained neural network model
            - vectorizer: The TF-IDF vectorizer
            - label_encoder: The label encoder for SKUs
            - metadata: Dictionary containing model metadata

    Raises:
        FileNotFoundError: If any of the model files are not found
        Exception: If there is an error loading the model
    """
    logger.info("Loading neural network model and preprocessing objects...")
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        metadata = joblib.load(METADATA_PATH)

        logger.info(
            f"Model covers {metadata['num_skus']} SKUs with at least {metadata['min_examples_per_sku']} examples each")
        logger.info(f"Test accuracy: {metadata['test_accuracy']:.3f}")

        return model, vectorizer, label_encoder, metadata
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# --- Prediction Functions ---


def predict_sku(description: str, maker: str = None, model: Any = None, vectorizer: Any = None, label_encoder: Any = None) -> Dict[str, Any]:
    """Predict the SKU for a given description.

    Args:
        description: The product description
        maker: The maker name (optional, used to filter predictions)
        model: The trained neural network model (optional, will be loaded if not provided)
        vectorizer: The TF-IDF vectorizer (optional, will be loaded if not provided)
        label_encoder: The label encoder for SKUs (optional, will be loaded if not provided)

    Returns:
        Dictionary containing:
            - top_sku: The most likely SKU
            - top_confidence: The confidence score for the top SKU
            - top_skus: List of dictionaries containing the top 5 SKUs and their confidence scores

    Raises:
        Exception: If there is an error making the prediction
    """
    try:
        # Load model if not provided
        if model is None or vectorizer is None or label_encoder is None:
            model, vectorizer, label_encoder, _ = load_model()

        # Add maker to description if provided
        if maker:
            # Ensure maker is in uppercase for consistency
            maker = maker.upper()
            # Add maker to the beginning of the description for better context
            full_description = f"{maker} {description}"
        else:
            full_description = description

        # Preprocess the description
        processed_desc = preprocess_text(full_description)
        logger.debug(f"Preprocessed description: '{processed_desc}'")

        # Extract features
        X = vectorizer.transform([processed_desc])

        # Predict SKU
        if hasattr(model, 'predict_proba'):
            # For models that support probability estimates
            probabilities = model.predict_proba(X)[0]

            # Get all SKUs with their probabilities
            all_skus = []
            for idx, prob in enumerate(probabilities):
                if prob > 0.001:  # Only consider SKUs with non-negligible probability
                    sku = label_encoder.inverse_transform([idx])[0]
                    all_skus.append({"sku": sku, "confidence": float(prob)})

            # Sort by confidence
            all_skus.sort(key=lambda x: x["confidence"], reverse=True)

            # Get the top 5 SKUs
            top_skus = all_skus[:5]

            # Get the top prediction
            top_sku = top_skus[0]["sku"]
            top_confidence = top_skus[0]["confidence"]
        else:
            # For models that don't support probability estimates
            top_sku = model.predict(X)[0]
            top_sku = label_encoder.inverse_transform([top_sku])[0]
            top_confidence = 1.0
            top_skus = [{"sku": top_sku, "confidence": top_confidence}]

        logger.info(
            f"Prediction for '{description}': {top_sku} with confidence {top_confidence:.3f}")

        return {
            "top_sku": top_sku,
            "top_confidence": top_confidence,
            "top_skus": top_skus
        }
    except Exception as e:
        logger.error(f"Error predicting SKU: {str(e)}", exc_info=True)
        raise

# --- Main Execution ---


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Test the prediction function
    try:
        # Load the model
        model, vectorizer, label_encoder, metadata = load_model()

        # Make a prediction
        description = "amortiguador delantero logan"
        maker = "RENAULT"
        result = predict_sku(description, maker, model,
                             vectorizer, label_encoder)

        # Make another prediction with a different maker
        description2 = "paragolpes delantero mazda 3"
        maker2 = "MAZDA"
        result2 = predict_sku(description2, maker2, model,
                              vectorizer, label_encoder)

        print(f"Description 1: {description} (Maker: {maker})")
        print(f"Predicted SKU: {result['top_sku']}")
        print(f"Confidence: {result['top_confidence']:.3f}")
        print("\nTop 5 SKUs:")
        for i, sku_info in enumerate(result['top_skus']):
            print(f"{i+1}. {sku_info['sku']}: {sku_info['confidence']:.3f}")

        print("\n---\n")

        print(f"Description 2: {description2} (Maker: {maker2})")
        print(f"Predicted SKU: {result2['top_sku']}")
        print(f"Confidence: {result2['top_confidence']:.3f}")
        print("\nTop 5 SKUs:")
        for i, sku_info in enumerate(result2['top_skus']):
            print(f"{i+1}. {sku_info['sku']}: {sku_info['confidence']:.3f}")
    except Exception as e:
        print(f"Error: {str(e)}")
