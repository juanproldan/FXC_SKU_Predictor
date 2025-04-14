# scripts/analyze_sku_description.py

import pandas as pd
import os
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_sku_description(input_csv_path: str, output_txt_path: str):
    """
    Analyzes the relationship between cleaned descriptions and SKUs in the data.
    Counts how many SKUs map to each description and saves summary statistics.
    """
    if not os.path.exists(input_csv_path):
        logging.error(f"Input file not found: {input_csv_path}")
        return

    logging.info(f"Loading data for analysis from: {input_csv_path}")
    try:
        # Read only necessary columns to save memory
        df = pd.read_csv(input_csv_path, usecols=['description_cleaned', 'SKU'])
        # Drop rows where either description_cleaned or SKU is missing
        df.dropna(subset=['description_cleaned', 'SKU'], inplace=True)
        logging.info(f"Loaded {len(df)} relevant rows.")
    except ValueError as e:
         logging.error(f"Column error loading CSV: {e}. Ensure 'description_cleaned' and 'SKU' exist.")
         return
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        return

    logging.info("Analyzing SKU counts per description...")
    # Group by description and count unique SKUs
    # Convert SKU to string to ensure proper grouping/counting if they are numeric
    sku_counts_per_desc = df.groupby('description_cleaned')['SKU'].nunique()

    total_unique_descriptions = len(sku_counts_per_desc)
    if total_unique_descriptions == 0:
        logging.warning("No descriptions found after filtering NAs. Cannot perform analysis.")
        return

    # Statistics
    descriptions_one_sku = (sku_counts_per_desc == 1).sum()
    descriptions_multiple_skus = (sku_counts_per_desc > 1).sum()
    max_skus_for_one_desc = sku_counts_per_desc.max()
    avg_skus_per_desc = sku_counts_per_desc.mean()

    percent_one_sku = (descriptions_one_sku / total_unique_descriptions) * 100
    percent_multiple_skus = (descriptions_multiple_skus / total_unique_descriptions) * 100

    # Find example descriptions mapping to many SKUs
    examples_multiple = sku_counts_per_desc[sku_counts_per_desc > 1].nlargest(5).index.tolist()

    # --- Prepare Results ---
    results = []
    results.append("--- SKU-Description Relationship Analysis ---")
    results.append(f"Total unique cleaned descriptions analyzed: {total_unique_descriptions}")
    results.append(f"\nDescriptions mapping to exactly one SKU: {descriptions_one_sku} ({percent_one_sku:.2f}%)")
    results.append(f"Descriptions mapping to multiple SKUs: {descriptions_multiple_skus} ({percent_multiple_skus:.2f}%)")
    results.append(f"\nMaximum SKUs mapped to a single description: {max_skus_for_one_desc}")
    results.append(f"Average SKUs per description: {avg_skus_per_desc:.2f}")
    results.append("\nTop 5 descriptions mapping to the most SKUs:")
    for i, desc in enumerate(examples_multiple, 1):
         count = sku_counts_per_desc[desc]
         results.append(f"  {i}. '{desc}' -> {count} SKUs")

    results.append("------------------------------------------\n")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_txt_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Save results to file
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(results))
        logging.info(f"Analysis results saved to: {output_txt_path}")
    except Exception as e:
        logging.error(f"Failed to write results to text file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SKU-description relationship in the cleaned data.")
    parser.add_argument(
        "--input",
        default="data/correlated_sku_data_cleaned.csv",
        help="Path to the input cleaned CSV file (default: data/correlated_sku_data_cleaned.csv)."
    )
    parser.add_argument(
        "--output",
        default="data/sku_description_analysis.txt",
        help="Path to save the output analysis text file (default: data/sku_description_analysis.txt)."
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

    analyze_sku_description(input_path, output_path)
