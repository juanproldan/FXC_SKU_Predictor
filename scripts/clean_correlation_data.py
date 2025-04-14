# scripts/clean_correlation_data.py

import pandas as pd
import os
import logging
import argparse
import re # Import regex module for cleaning

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    """Applies basic text cleaning: lowercase, strip whitespace, reduce multiple spaces."""
    if pd.isna(text):
        return text
    text = str(text).lower()  # Convert to lowercase
    text = text.strip()       # Remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single space
    return text

def clean_data(input_csv_path: str, output_csv_path: str):
    """
    Loads the correlated data CSV, cleans the 'description' column,
    and saves the result to a new CSV file.
    """
    if not os.path.exists(input_csv_path):
        logging.error(f"Input file not found: {input_csv_path}")
        return

    logging.info(f"Loading data for cleaning from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
        logging.info(f"Loaded {len(df)} rows.")
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        return

    # --- Data Cleaning ---
    if 'description' in df.columns:
        initial_descriptions = df['description'].nunique()
        logging.info(f"Cleaning 'description' column (Initial unique: {initial_descriptions})...")
        # Apply the cleaning function
        df['description_cleaned'] = df['description'].apply(clean_text)

        # Drop original description column? Or keep both? Keep both for now.
        # df = df.drop(columns=['description'])
        # df = df.rename(columns={'description_cleaned': 'description'})

        final_descriptions = df['description_cleaned'].nunique()
        logging.info(f"Cleaning complete. Final unique descriptions: {final_descriptions}")
    else:
        logging.warning("Column 'description' not found. No cleaning applied.")

    # --- Saving Cleaned Data ---
    logging.info(f"Saving cleaned data to: {output_csv_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    try:
        # Save to CSV
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        logging.info("Cleaned data saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save cleaned CSV file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the 'description' column in the correlated SKU data CSV.")
    parser.add_argument(
        "--input",
        default="data/correlated_sku_data.csv",
        help="Path to the input CSV file (default: data/correlated_sku_data.csv)."
    )
    parser.add_argument(
        "--output",
        default="data/correlated_sku_data_cleaned.csv",
        help="Path to save the cleaned output CSV file (default: data/correlated_sku_data_cleaned.csv)."
    )

    args = parser.parse_args()

    # Determine project root assuming script is in 'scripts' subdir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Construct absolute paths relative to project root if needed
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(project_root, input_path)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)

    clean_data(input_path, output_path)
