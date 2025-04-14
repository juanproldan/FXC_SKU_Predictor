import pandas as pd
import re
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Placeholder Dictionaries ---
# In a real scenario, these would likely be loaded from config or external files
SPELLING_CORRECTIONS: Dict[str, str] = {
    r'\bcapot\b': 'capó',  # Example: correct 'capot' to 'capó'
    r'\bparagolp\b': 'paragolpes', # Example
    # Add more regex-based corrections as needed
}

TERM_STANDARDIZATION: Dict[str, str] = {
    r'\bcofre\b': 'capó', # Example: standardize 'cofre' to 'capó'
    r'\bdefensa\b': 'paragolpes', # Example
    # Add more regex-based standardizations as needed
}

# --- Preprocessing Functions ---

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove extra whitespace."""
    if not isinstance(text, str):
        logging.warning(f"Expected string but got {type(text)}. Returning empty string.")
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single, trim ends
    # Add more cleaning steps if needed (e.g., remove punctuation, numbers)
    # text = re.sub(r'[^\w\s]', '', text) # Example: Remove punctuation
    return text

def correct_spelling(text: str, corrections: Dict[str, str]) -> str:
    """Applies spelling corrections based on a dictionary of regex patterns."""
    if not isinstance(text, str):
        return ""
    for गलत, सही in corrections.items(): # Using Spanish words for fun: wrong, correct
        try:
            text = re.sub(गलत, सही, text, flags=re.IGNORECASE)
        except re.error as e:
            logging.warning(f"Regex error in spelling correction pattern '{गलत}': {e}")
    return text

def standardize_terms(text: str, standardization_map: Dict[str, str]) -> str:
    """Standardizes terms based on a dictionary of regex patterns."""
    if not isinstance(text, str):
        return ""
    for term, standard_term in standardization_map.items():
        try:
            text = re.sub(term, standard_term, text, flags=re.IGNORECASE)
        except re.error as e:
            logging.warning(f"Regex error in standardization pattern '{term}': {e}")
    return text

def preprocess_descriptions(df: pd.DataFrame, description_col: str = 'description') -> pd.DataFrame:
    """Applies the full preprocessing pipeline to the description column."""
    if description_col not in df.columns:
        logging.error(f"Description column '{description_col}' not found in DataFrame.")
        raise ValueError(f"Description column '{description_col}' not found.")

    # Fill NaNs in description column with empty strings to avoid errors
    df[description_col] = df[description_col].fillna('')

    logging.info(f"Starting preprocessing for column '{description_col}'...")

    # Apply cleaning
    df[f'{description_col}_processed'] = df[description_col].apply(clean_text)
    logging.info("Applied basic text cleaning.")

    # Apply spelling correction
    df[f'{description_col}_processed'] = df[f'{description_col}_processed'].apply(
        lambda x: correct_spelling(x, SPELLING_CORRECTIONS)
    )
    logging.info("Applied spelling correction.")

    # Apply term standardization
    df[f'{description_col}_processed'] = df[f'{description_col}_processed'].apply(
        lambda x: standardize_terms(x, TERM_STANDARDIZATION)
    )
    logging.info("Applied term standardization.")

    logging.info("Preprocessing completed.")
    return df

# --- Main execution for testing ---
if __name__ == '__main__':
    # Create a sample DataFrame for testing
    sample_data = {
        'description': [
            "Capot para Coche XY",
            " Paragolp delantero",
            "  Cofre con golpe   ",
            "Defensa trasera",
            "Faro Izquierdo Roto",
            None, # Test NaN handling
            12345 # Test non-string handling
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