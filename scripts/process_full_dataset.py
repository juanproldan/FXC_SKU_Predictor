import json
import csv
import os
import sys
import traceback
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_full_dataset.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Input/output paths
INPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'Consolidado.json')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'full_training_data.csv')

REQUIRED_ITEM_FIELDS = ["referencia", "descripcion"]
REQUIRED_ENTRY_FIELDS = ["maker", "series", "model"]

def process_json_to_csv(input_path, output_path):
    """
    Process the entire JSON file and convert it to CSV format.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to the output CSV file
    """
    logger.info(f"Processing JSON file: {input_path}")
    
    try:
        # Load the JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            logger.info("Loading JSON data...")
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} entries from JSON file")
        
        # Process the data
        rows = []
        skipped_entries = 0
        skipped_items = 0
        
        logger.info("Processing entries...")
        for entry_idx, entry in enumerate(tqdm(data, desc="Processing entries")):
            # Extract entry-level fields
            maker = entry.get("maker", "")
            series = entry.get("series", "")
            model = entry.get("model", entry.get("fabrication_year", ""))
            
            # Skip entries missing required fields
            if not all([maker, series, model]):
                skipped_entries += 1
                continue
            
            # Process items within the entry
            items = entry.get("items", [])
            if not items:
                skipped_entries += 1
                continue
            
            for item in items:
                referencia = item.get("referencia", "")
                descripcion = item.get("descripcion", "")
                
                # Skip items missing required fields
                if not all([referencia, descripcion]):
                    skipped_items += 1
                    continue
                
                # Add the row
                rows.append({
                    "maker": maker,
                    "series": series,
                    "model": model,
                    "descripcion": descripcion,
                    "referencia": referencia
                })
            
            # Log progress periodically
            if (entry_idx + 1) % 1000 == 0:
                logger.info(f"Processed {entry_idx + 1} entries, generated {len(rows)} rows so far")
        
        logger.info(f"Processing complete. Generated {len(rows)} rows")
        logger.info(f"Skipped {skipped_entries} entries and {skipped_items} items due to missing required fields")
        
        # Write the CSV file
        logger.info(f"Writing CSV file: {output_path}")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["maker", "series", "model", "descripcion", "referencia"], quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Successfully exported {len(rows)} rows to {output_path}")
        
        # Generate some basic statistics
        maker_counts = {}
        sku_counts = {}
        
        for row in rows:
            maker = row["maker"]
            sku = row["referencia"]
            
            maker_counts[maker] = maker_counts.get(maker, 0) + 1
            sku_counts[sku] = sku_counts.get(sku, 0) + 1
        
        logger.info(f"Dataset contains {len(maker_counts)} unique makers")
        logger.info(f"Dataset contains {len(sku_counts)} unique SKUs")
        
        # Log the top 10 makers by count
        top_makers = sorted(maker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 makers by count:")
        for maker, count in top_makers:
            logger.info(f"  {maker}: {count} rows")
        
        # Count SKUs with different frequencies
        sku_freq = {
            "1": sum(1 for count in sku_counts.values() if count == 1),
            "2-5": sum(1 for count in sku_counts.values() if 2 <= count <= 5),
            "6-10": sum(1 for count in sku_counts.values() if 6 <= count <= 10),
            "11+": sum(1 for count in sku_counts.values() if count > 10)
        }
        
        logger.info("SKU frequency distribution:")
        for freq, count in sku_freq.items():
            percentage = count / len(sku_counts) * 100
            logger.info(f"  SKUs with {freq} occurrences: {count} ({percentage:.1f}%)")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing JSON file: {str(e)}")
        traceback.print_exc()
        return False

def main():
    try:
        logger.info("Starting full dataset processing")
        
        # Check if input file exists
        if not os.path.exists(INPUT_PATH):
            logger.error(f"Input file not found: {INPUT_PATH}")
            return
        
        # Process the data
        success = process_json_to_csv(INPUT_PATH, OUTPUT_PATH)
        
        if success:
            logger.info("Processing completed successfully")
        else:
            logger.error("Processing failed")
    
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
