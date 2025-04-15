# scripts/create_correlation_data.py

import ijson
import pandas as pd
import os
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_correlation_data(input_json_path: str, output_csv_path: str, checkpoint_interval: int = 1000):
    """
    Loads a JSON file containing records with 'items' lists,
    extracts relevant fields (maker, series, model, SKU, description),
    and saves the correlated data to a CSV file.
    Saves progress every `checkpoint_interval` records to a .partial CSV file.
    """
    import json
    import sys

    partial_csv_path = output_csv_path + '.partial'
    correlated_data = []
    record_count = 0
    item_count = 0
    resume_from_partial = False

    # Check for partial file
    if os.path.exists(partial_csv_path):
        response = input(f"A partial checkpoint file was found at {partial_csv_path}. Resume from checkpoint? (y/n): ")
        if response.strip().lower() == 'y':
            try:
                correlated_data = pd.read_csv(partial_csv_path).to_dict('records')
                record_count = len(correlated_data)
                logging.info(f"Resuming from partial checkpoint: {record_count} records loaded.")
                resume_from_partial = True
            except Exception as e:
                logging.error(f"Failed to load partial checkpoint: {e}. Starting from scratch.")
                correlated_data = []
                record_count = 0

    if not os.path.exists(input_json_path):
        logging.error(f"Input file not found: {input_json_path}")
        return

    logging.info(f"Loading data from: {input_json_path}")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            records = ijson.items(f, 'item')
            for record in records:
                record_count += 1
                maker = record.get('maker')
                series = record.get('series')
                model = record.get('model')
                items = record.get('items', [])
                if not isinstance(items, list):
                    logging.warning(f"Record {record_count}: 'items' is not a list. Skipping.")
                    continue
                for item in items:
                    item_count += 1
                    sku = item.get('referencia')
                    descripcion = item.get('descripcion')
                    correlated_data.append({
                        'maker': maker,
                        'series': series,
                        'model': model,
                        'SKU': sku,
                        'description': descripcion
                    })
                # Save checkpoint every N records
                if record_count % checkpoint_interval == 0:
                    try:
                        df_partial = pd.DataFrame(correlated_data)
                        df_partial.to_csv(partial_csv_path, index=False, encoding='utf-8')
                        logging.info(f"Checkpoint saved at {record_count} records to {partial_csv_path}.")
                    except Exception as e:
                        logging.error(f"Failed to save checkpoint: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON file: {e}")
        return
    except Exception as e:
        logging.error(f"Failed to load JSON file: {e}")
        return

    if not correlated_data:
        logging.warning("No items found in any records. Output file will not be created.")
        return

    logging.info(f"Processed {record_count} records and extracted {item_count} items.")

    # Create DataFrame
    df = pd.DataFrame(correlated_data)

    # Optional: Clean up data (e.g., remove rows with missing SKU or description)
    df.dropna(subset=['SKU', 'description'], inplace=True)
    logging.info(f"DataFrame shape after creation: {df.shape}")
    logging.info(f"Saving correlated data to: {output_csv_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Save to CSV
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        logging.info("Correlated data saved successfully.")
        # Remove partial file if it exists
        if os.path.exists(partial_csv_path):
            os.remove(partial_csv_path)
            logging.info(f"Removed checkpoint file: {partial_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create correlated SKU data CSV from JSON.")
    parser.add_argument(
        "--input",
        default="C:\\Users\\juanp\\OneDrive\\Documents\\Python\\0_Training\\017_Fixacar\\Consolidado.json",
        help="Path to the input JSON file (default: data/items_not_empty.json)."
    )
    parser.add_argument(
        "--output",
        default="data/correlated_sku_data.csv",
        help="Path to save the output CSV file (default: data/correlated_sku_data.csv)."
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Number of records to process before saving a checkpoint (default: 1000)."
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

    create_correlation_data(input_path, output_path, checkpoint_interval=args.checkpoint_interval)
