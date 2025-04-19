#!/usr/bin/env python
"""
Script to create detailed CSV files by maker from Consolidado.json.

This script processes the Consolidado.json file and creates a separate CSV file
for each maker, containing all the details for each SKU (one SKU per row).
"""

import json
import csv
import os
import sys
import traceback
import logging
from tqdm import tqdm
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("json_to_maker_csv_detailed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Input/output paths
INPUT_PATH = r"C:\Users\juanp\OneDrive\Documents\Python\0_Training\017_Fixacar\001_SKU_desde_descripcion\Fuente_Json_Consolidado\Consolidado.json"
OUTPUT_DIR = r"C:\Users\juanp\OneDrive\Documents\Python\0_Training\017_Fixacar"

# Create output directory for maker files
MAKER_DIR = os.path.join(OUTPUT_DIR, "maker_csv_detailed")
os.makedirs(MAKER_DIR, exist_ok=True)

# Fields to include in the CSV files
CSV_FIELDS = ["maker", "series", "model", "descripcion", "referencia", "precio", "cantidad", "estado"]

def clean_maker_name(maker):
    """
    Clean and normalize maker name for use in filenames.
    
    Args:
        maker: The maker name to clean
        
    Returns:
        A cleaned version of the maker name suitable for use in filenames
    """
    # Convert to lowercase
    maker = maker.lower()
    
    # Replace spaces and special characters
    maker = maker.replace(' ', '_')
    maker = ''.join(c for c in maker if c.isalnum() or c == '_')
    
    return maker

def process_json_by_maker(input_path, output_dir):
    """
    Process the JSON file and create separate CSV files for each maker.
    
    Args:
        input_path: Path to the input JSON file
        output_dir: Directory to save the output CSV files
    """
    logger.info(f"Processing JSON file: {input_path}")
    
    try:
        # Load the JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            logger.info("Loading JSON data...")
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} entries from JSON file")
        
        # Dictionary to store rows by maker
        maker_rows = {}
        
        # Statistics
        skipped_entries = 0
        skipped_items = 0
        empty_skus = 0
        
        # Process the data
        logger.info("Processing entries...")
        for entry_idx, entry in enumerate(tqdm(data, desc="Processing entries")):
            # Extract entry-level fields
            maker = entry.get("maker", "")
            series = entry.get("series", "")
            model = entry.get("model", entry.get("fabrication_year", ""))
            
            # Skip entries missing required fields
            if not maker:
                skipped_entries += 1
                continue
            
            # Process items within the entry
            items = entry.get("items", [])
            if not items:
                skipped_entries += 1
                continue
            
            # Initialize maker in dictionary if not exists
            if maker not in maker_rows:
                maker_rows[maker] = []
            
            for item in items:
                referencia = item.get("referencia", "")
                descripcion = item.get("descripcion", "")
                
                # Skip items with empty SKUs
                if not referencia or referencia.strip() == "":
                    empty_skus += 1
                    continue
                
                # Extract additional fields
                precio = item.get("precio", "")
                cantidad = item.get("cantidad", "")
                estado = item.get("estado", "")
                
                # Add the row to the maker's list
                maker_rows[maker].append({
                    "maker": maker,
                    "series": series,
                    "model": model,
                    "descripcion": descripcion,
                    "referencia": referencia,
                    "precio": precio,
                    "cantidad": cantidad,
                    "estado": estado
                })
            
            # Log progress periodically
            if (entry_idx + 1) % 10000 == 0:
                total_rows = sum(len(rows) for rows in maker_rows.values())
                logger.info(f"Processed {entry_idx + 1} entries, generated {total_rows} rows so far")
        
        # Write CSV files for each maker
        logger.info(f"Writing CSV files to {output_dir}")
        
        # Create a summary file
        summary_path = os.path.join(output_dir, "maker_csv_summary.csv")
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["Maker", "Number of Items"])
            
            for maker, rows in sorted(maker_rows.items()):
                # Skip makers with no rows
                if not rows:
                    continue
                
                # Clean maker name for filename
                clean_name = clean_maker_name(maker)
                
                # Create CSV file for this maker
                maker_path = os.path.join(output_dir, f"{clean_name}_detailed.csv")
                
                with open(maker_path, 'w', newline='', encoding='utf-8') as mf:
                    writer_maker = csv.DictWriter(mf, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
                    writer_maker.writeheader()
                    writer_maker.writerows(rows)
                
                logger.info(f"Created CSV for {maker} with {len(rows)} rows: {maker_path}")
                
                # Add to summary
                writer.writerow([maker, len(rows)])
        
        # Generate overall statistics
        total_rows = sum(len(rows) for rows in maker_rows.values())
        logger.info(f"Processing complete. Generated {total_rows} rows across {len(maker_rows)} makers")
        logger.info(f"Skipped {skipped_entries} entries due to missing required fields")
        logger.info(f"Skipped {empty_skus} items due to empty SKUs")
        
        # Log the top 10 makers by count
        maker_counts = {maker: len(rows) for maker, rows in maker_rows.items()}
        top_makers = sorted(maker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 makers by count:")
        for maker, count in top_makers:
            logger.info(f"  {maker}: {count} rows")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing JSON file: {str(e)}")
        traceback.print_exc()
        return False

def main():
    try:
        logger.info("Starting JSON to detailed maker CSV conversion")
        
        # Check if input file exists
        if not os.path.exists(INPUT_PATH):
            logger.error(f"Input file not found: {INPUT_PATH}")
            return
        
        # Process the data
        success = process_json_by_maker(INPUT_PATH, MAKER_DIR)
        
        if success:
            logger.info(f"Processing completed successfully. CSV files saved to {MAKER_DIR}")
        else:
            logger.error("Processing failed")
    
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
