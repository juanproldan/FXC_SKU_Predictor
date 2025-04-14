# scripts/extract_non_empty_items.py

import pandas as pd
import json
import os
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_records_with_items(input_json_path: str, output_json_path: str):
    """
    Loads a JSON file, filters records where the 'items' field is a non-empty list,
    and saves the filtered records to a new JSON file.
    """
    if not os.path.exists(input_json_path):
        logging.error(f"Input file not found: {input_json_path}")
        return

    logging.info(f"Loading data from: {input_json_path}")
    try:
        # Load the entire JSON into a pandas DataFrame
        df = pd.read_json(input_json_path)
        logging.info(f"Loaded {len(df)} records.")
    except Exception as e:
        logging.error(f"Failed to load JSON file: {e}")
        return

    # --- Filtering Logic ---
    # 1. Check if 'items' column exists
    if 'items' not in df.columns:
        logging.error("'items' column not found in the JSON data.")
        return

    # 2. Filter out rows where 'items' is null/NaN
    initial_count = len(df)
    df_filtered = df.dropna(subset=['items'])
    dropped_nan = initial_count - len(df_filtered)
    if dropped_nan > 0:
        logging.info(f"Removed {dropped_nan} records where 'items' was null.")

    # 3. Filter out rows where 'items' is not a list
    initial_count = len(df_filtered)
    is_list_mask = df_filtered['items'].apply(lambda x: isinstance(x, list))
    df_filtered = df_filtered[is_list_mask]
    dropped_non_list = initial_count - len(df_filtered)
    if dropped_non_list > 0:
        logging.info(f"Removed {dropped_non_list} records where 'items' was not a list.")

    # 4. Filter out rows where the 'items' list is empty
    initial_count = len(df_filtered)
    non_empty_list_mask = df_filtered['items'].apply(lambda x: len(x) > 0)
    df_filtered = df_filtered[non_empty_list_mask]
    dropped_empty_list = initial_count - len(df_filtered)
    if dropped_empty_list > 0:
        logging.info(f"Removed {dropped_empty_list} records where 'items' list was empty.")
    # ------------------------

    if df_filtered.empty:
        logging.warning("No records found with non-empty 'items'. Output file will not be created.")
        return

    logging.info(f"Found {len(df_filtered)} records with non-empty 'items'.")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    logging.info(f"Saving filtered records to: {output_json_path}")
    try:
        # Save the filtered DataFrame back to JSON
        # Use orient='records' for a list of records format
        # Use lines=True for JSON Lines format (one JSON object per line, often better for large files)
        df_filtered.to_json(output_json_path, orient='records', lines=False, force_ascii=False, indent=2)
        logging.info("Filtered records saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save filtered JSON: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract records with non-empty 'items' from a JSON file.")
    parser.add_argument(
        "--input",
        default="Consolidado.json",
        help="Path to the input JSON file (default: Consolidado.json in project root)."
    )
    parser.add_argument(
        "--output",
        default="data/items_not_empty.json",
        help="Path to save the filtered output JSON file (default: data/items_not_empty.json)."
    )

    args = parser.parse_args()

    # Construct full paths relative to the project root if needed
    # This assumes the script is run from the project root or the paths are absolute
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    # Correctly determine project root assuming script is in 'scripts' subdir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Recalculate paths relative to project root if they aren't absolute
    if not os.path.isabs(args.input):
        input_path = os.path.join(project_root, args.input)
    if not os.path.isabs(args.output):
        output_path = os.path.join(project_root, args.output)


    extract_records_with_items(input_path, output_path)
