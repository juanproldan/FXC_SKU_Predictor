"""
Text preprocessing utilities for the SKU prediction system.

This module provides functions for cleaning and standardizing text data
for use in the SKU prediction system.
"""

import pandas as pd
import re
import logging
import string
from typing import Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# --- Placeholder Dictionaries ---
# In a real scenario, these would likely be loaded from config or external files
SPELLING_CORRECTIONS: Dict[str, str] = {
    r'\bcapot\b': 'capó',  # Example: correct 'capot' to 'capó'
    r'\bparagolp\b': 'paragolpes',  # Example
    # Add more regex-based corrections as needed
}

TERM_STANDARDIZATION: Dict[str, str] = {
    r'\bcofre\b': 'capó',  # Example: standardize 'cofre' to 'capó'
    r'\bdefensa\b': 'paragolpes',  # Example
    # Add more regex-based standardizations as needed
}

# --- Preprocessing Functions ---

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove extra whitespace, punctuation.
    
    Args:
        text: The text to clean.
        
    Returns:
        The cleaned text.
    """
    if not isinstance(text, str):
        # Log or handle non-string types appropriately
        logger.warning(f"Expected string but got {type(text)}. Returning empty string.")
        return ""  # Return empty string for non-strings
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Replace multiple spaces with single, trim ends
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def correct_spelling(text: str, corrections: Dict[str, str] = None) -> str:
    """Applies spelling corrections based on a dictionary of regex patterns.
    
    Args:
        text: The text to correct.
        corrections: A dictionary mapping regex patterns to corrections.
            If None, uses the default SPELLING_CORRECTIONS.
            
    Returns:
        The text with spelling corrections applied.
    """
    if not isinstance(text, str):
        return ""
    
    if corrections is None:
        corrections = SPELLING_CORRECTIONS
    
    for pattern, correction in corrections.items():
        try:
            text = re.sub(pattern, correction, text, flags=re.IGNORECASE)
        except re.error as e:
            logger.warning(f"Regex error in spelling correction pattern '{pattern}': {e}")
    
    return text

def standardize_terms(text: str, standardization_map: Dict[str, str] = None) -> str:
    """Standardizes terms based on a dictionary of regex patterns.
    
    Args:
        text: The text to standardize.
        standardization_map: A dictionary mapping regex patterns to standard terms.
            If None, uses the default TERM_STANDARDIZATION.
            
    Returns:
        The text with standardized terms.
    """
    if not isinstance(text, str):
        return ""
    
    if standardization_map is None:
        standardization_map = TERM_STANDARDIZATION
    
    for term, standard_term in standardization_map.items():
        try:
            text = re.sub(term, standard_term, text, flags=re.IGNORECASE)
        except re.error as e:
            logger.warning(f"Regex error in standardization pattern '{term}': {e}")
    
    return text

def preprocess_text(text: str) -> str:
    """Apply the full preprocessing pipeline to a single text string.
    
    Args:
        text: The text to preprocess.
        
    Returns:
        The preprocessed text.
    """
    text = clean_text(text)
    text = correct_spelling(text)
    text = standardize_terms(text)
    return text

def preprocess_descriptions(df: pd.DataFrame, description_col: str) -> pd.DataFrame:
    """Applies the full preprocessing pipeline to the description column.
    
    Args:
        df: The DataFrame containing the description column.
        description_col: The name of the description column.
        
    Returns:
        The DataFrame with an additional column containing the preprocessed descriptions.
        
    Raises:
        ValueError: If the description column is not found in the DataFrame.
    """
    # Check if the column exists *first*
    if description_col not in df.columns:
        logger.error(f"Description column '{description_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        raise ValueError(f"Description column '{description_col}' not found.")

    # Fill NaNs in description column with empty strings to avoid errors
    df[description_col] = df[description_col].fillna('')

    logger.info(f"Starting preprocessing for column '{description_col}'...")

    # Apply cleaning
    logger.info(f"Attempting basic text cleaning on column: '{description_col}'")
    try:
        df[f'{description_col}_processed'] = df[description_col].apply(clean_text)
        logger.info("Applied basic text cleaning successfully.")
    except Exception as e:
        logger.error(f"Error during basic text cleaning: {e}", exc_info=True)
        raise

    # Apply spelling correction
    logger.info(f"Attempting spelling correction on column: '{description_col}_processed'")
    try:
        df[f'{description_col}_processed'] = df[f'{description_col}_processed'].apply(
            lambda x: correct_spelling(x, SPELLING_CORRECTIONS)
        )
        logger.info("Applied spelling correction successfully.")
    except Exception as e:
        logger.error(f"Error during spelling correction: {e}", exc_info=True)
        raise

    # Apply term standardization
    logger.info(f"Attempting term standardization on column: '{description_col}_processed'")
    try:
        df[f'{description_col}_processed'] = df[f'{description_col}_processed'].apply(
            lambda x: standardize_terms(x, TERM_STANDARDIZATION)
        )
        logger.info("Applied term standardization successfully.")
    except Exception as e:
        logger.error(f"Error during term standardization: {e}", exc_info=True)
        raise

    logger.info("Preprocessing completed.")
    return df

# --- Main execution for testing ---
if __name__ == '__main__':
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a sample DataFrame for testing
    sample_data = {
        'description': [
            "Capot para Coche XY",
            " Paragolp delantero",
            "  Cofre con golpe   ",
            "Defensa trasera",
            "Faro Izquierdo Roto",
            None,  # Test NaN handling
            12345  # Test non-string handling
        ],
        'sku': ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005', 'SKU006', 'SKU007']
    }
    sample_df = pd.DataFrame(sample_data)

    print("--- Original Data ---")
    print(sample_df)

    # Preprocess the descriptions
    try:
        processed_df = preprocess_descriptions(sample_df.copy(), description_col='description')
        print("\n--- Processed Data ---")
        print(processed_df[['description', 'description_processed']])
    except ValueError as e:
        print(f"\nError during preprocessing: {e}")

    # Example of accessing correction dictionaries (could be loaded from config)
    print(f"\nSpelling Corrections Used: {SPELLING_CORRECTIONS}")
    print(f"Term Standardizations Used: {TERM_STANDARDIZATION}")
