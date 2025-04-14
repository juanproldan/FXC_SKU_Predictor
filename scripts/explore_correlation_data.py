# scripts/explore_correlation_data.py

import pandas as pd
import os
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def explore_data(input_csv_path: str, output_txt_path: str):
    """
    Loads the correlated data CSV, calculates basic statistics,
    and saves them to a text file.
    """
    if not os.path.exists(input_csv_path):
        logging.error(f"Input file not found: {input_csv_path}")
        return

    logging.info(f"Loading data from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
        logging.info(f"Loaded {len(df)} rows.")
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        return

    # --- Exploration ---
    results = []
    results.append("--- Data Exploration Results ---")
    results.append(f"Total number of rows (items): {len(df)}")

    # Check if columns exist before calculating nunique
    columns_to_check = ['SKU', 'description', 'maker', 'series', 'model']
    for col in columns_to_check:
        if col in df.columns:
            unique_count = df[col].nunique()
            results.append(f"Number of unique '{col}': {unique_count}")
        else:
            logging.warning(f"Column '{col}' not found in the CSV.")
            results.append(f"Column '{col}' not found.")

    results.append("------------------------------\n")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_txt_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Save results to file
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(results))
        logging.info(f"Exploration results saved to: {output_txt_path}")
    except Exception as e:
        logging.error(f"Failed to write results to text file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore the correlated SKU data CSV.")
    parser.add_argument(
        "--input",
        default="data/correlated_sku_data.csv",
        help="Path to the input CSV file (default: data/correlated_sku_data.csv)."
    )
    parser.add_argument(
        "--output",
        default="data/exploration_summary.txt",
        help="Path to save the output summary text file (default: data/exploration_summary.txt)."
    )

    args = parser.parse_args()

    # Determine project root assuming script is in 'scripts' subdir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Construct absolute path relative to project root if needed
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(project_root, input_path)
    
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)

    explore_data(input_path, output_path)
