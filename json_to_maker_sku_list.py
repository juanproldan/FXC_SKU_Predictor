#!/usr/bin/env python
"""
Script to extract SKUs from Consolidado.json and create separate text files by maker.

This script processes the Consolidado.json file and creates a separate text file
for each maker, containing just the SKUs (one per line).
"""

import json
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
        logging.FileHandler("json_to_maker_sku_list.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Input/output paths
INPUT_PATH = r"C:\Users\juanp\OneDrive\Documents\Python\0_Training\017_Fixacar\001_SKU_desde_descripcion\Fuente_Json_Consolidado\Consolidado.json"
OUTPUT_DIR = r"C:\Users\juanp\OneDrive\Documents\Python\0_Training\017_Fixacar"

# Create output directory for maker files
MAKER_DIR = os.path.join(OUTPUT_DIR, "maker_skus")
os.makedirs(MAKER_DIR, exist_ok=True)

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
    Process the JSON file and create separate text files with SKUs for each maker.
    
    Args:
        input_path: Path to the input JSON file
        output_dir: Directory to save the output text files
    """
    logger.info(f"Processing JSON file: {input_path}")
    
    try:
        # Load the JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            logger.info("Loading JSON data...")
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} entries from JSON file")
        
        # Dictionary to store SKUs by maker
        maker_skus = {}
        
        # Statistics
        skipped_entries = 0
        skipped_items = 0
        empty_skus = 0
        
        # Process the data
        logger.info("Processing entries...")
        for entry_idx, entry in enumerate(tqdm(data, desc="Processing entries")):
            # Extract maker
            maker = entry.get("maker", "")
            
            # Skip entries with no maker
            if not maker:
                skipped_entries += 1
                continue
            
            # Process items within the entry
            items = entry.get("items", [])
            if not items:
                skipped_entries += 1
                continue
            
            # Initialize maker in dictionary if not exists
            if maker not in maker_skus:
                maker_skus[maker] = set()  # Use a set to avoid duplicates
            
            for item in items:
                referencia = item.get("referencia", "")
                
                # Skip items with empty SKUs
                if not referencia or referencia.strip() == "":
                    empty_skus += 1
                    continue
                
                # Add the SKU to the maker's set
                maker_skus[maker].add(referencia.strip())
            
            # Log progress periodically
            if (entry_idx + 1) % 10000 == 0:
                total_skus = sum(len(skus) for skus in maker_skus.values())
                logger.info(f"Processed {entry_idx + 1} entries, found {total_skus} unique SKUs so far")
        
        # Write text files for each maker
        logger.info(f"Writing SKU files to {output_dir}")
        
        # Create a summary file
        summary_path = os.path.join(output_dir, "maker_sku_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Maker\tNumber of Unique SKUs\n")
            
            for maker, skus in sorted(maker_skus.items()):
                # Skip makers with no SKUs
                if not skus:
                    continue
                
                # Clean maker name for filename
                clean_name = clean_maker_name(maker)
                
                # Create text file for this maker
                maker_path = os.path.join(output_dir, f"{clean_name}_skus.txt")
                
                with open(maker_path, 'w', encoding='utf-8') as mf:
                    # Write each SKU on a separate line
                    for sku in sorted(skus):
                        mf.write(f"{sku}\n")
                
                logger.info(f"Created SKU list for {maker} with {len(skus)} unique SKUs: {maker_path}")
                
                # Add to summary
                f.write(f"{maker}\t{len(skus)}\n")
        
        # Generate overall statistics
        total_skus = sum(len(skus) for skus in maker_skus.values())
        logger.info(f"Processing complete. Found {total_skus} unique SKUs across {len(maker_skus)} makers")
        logger.info(f"Skipped {skipped_entries} entries due to missing required fields")
        logger.info(f"Skipped {empty_skus} items due to empty SKUs")
        
        # Log the top 10 makers by count
        maker_counts = {maker: len(skus) for maker, skus in maker_skus.items()}
        top_makers = sorted(maker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 makers by unique SKU count:")
        for maker, count in top_makers:
            logger.info(f"  {maker}: {count} unique SKUs")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing JSON file: {str(e)}")
        traceback.print_exc()
        return False

def main():
    try:
        logger.info("Starting JSON to maker SKU list conversion")
        
        # Check if input file exists
        if not os.path.exists(INPUT_PATH):
            logger.error(f"Input file not found: {INPUT_PATH}")
            return
        
        # Process the data
        success = process_json_by_maker(INPUT_PATH, MAKER_DIR)
        
        if success:
            logger.info(f"Processing completed successfully. SKU lists saved to {MAKER_DIR}")
        else:
            logger.error("Processing failed")
    
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
