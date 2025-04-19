import logging
import time
import os
import pandas as pd

# Configure logging
# Set up logging to file and console
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
log_file_path = os.path.join(log_directory, 'sku_prediction.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path), # Log to file
        logging.StreamHandler() # Log to console
    ]
)

logger = logging.getLogger(__name__)

# Import project modules after logging is configured
from src.config_loader import load_config
from src.data_loader import load_data
from src.preprocessing import preprocess_descriptions

def main():
    """Main function to run the SKU prediction pipeline."""
    start_time = time.time()
    logger.info("Starting the SKU prediction pipeline...")

    try:
        # 1. Load Configuration
        logger.info("Loading configuration...")
        config = load_config()
        if not config:
            logger.error("Failed to load configuration. Exiting.")
            return
        logger.info("Configuration loaded successfully.")

        # 2. Load Data
        logger.info("Loading data...")
        data_df = load_data()
        if data_df is None:
            logger.error("Failed to load data. Exiting.")
            return
        logger.info(f"Data loaded successfully. Shape: {data_df.shape}")
        logger.info(f"DataFrame Columns: {data_df.columns.tolist()}")
        # Optional: Display initial data info
        # logger.debug("\nInitial Data Head:\n%s", data_df.head().to_string())
        # logger.debug("\nInitial Data Info:")
        # data_df.info(buf=logger.info) # Redirect info() output to logger

        # 3. Preprocess Data
        # Use the 'descripcion' column from the flattened data
        description_col_actual = 'descripcion'
        if description_col_actual not in data_df.columns:
            logger.error(f"Description column '{description_col_actual}' not found in the final DataFrame. Cannot preprocess descriptions.")
            logger.info(f"Final DataFrame Columns: {data_df.columns.tolist()}")
            return # Stop if no description column is found

        logger.info(f"Starting data preprocessing for column: '{description_col_actual}'...")
        processed_df = preprocess_descriptions(data_df.copy(), description_col=description_col_actual)
        logger.info(f"Data preprocessing completed. New shape: {processed_df.shape}")
        # Adjust the name of the processed column based on the actual input column
        processed_col_name = f'{description_col_actual}_processed'
        logger.info(f"Processed description column added: '{processed_col_name}'")

        # Optional: Display processed data info
        # logger.debug("\nProcessed Data Head (showing relevant columns):\n%s",
        #              processed_df[[description_col_actual, processed_col_name]].head().to_string())

        # Optional: Save Processed Data
        processed_data_path = config.get('data', {}).get('processed_data_path')
        if processed_data_path:
            try:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
                logger.info(f"Saving processed data to: {processed_data_path}...")
                # Use appropriate format, e.g., CSV or Parquet for efficiency
                processed_df.to_csv(processed_data_path, index=False)
                logger.info("Processed data saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save processed data to {processed_data_path}: {e}")
        else:
            logger.warning("Processed data path not specified in config. Skipping saving.")

        # --- Placeholder for next steps ---
        # 4. Feature Engineering
        logger.info("Skipping Feature Engineering (Not implemented yet)...")

        # 5. Model Training
        logger.info("Skipping Model Training (Not implemented yet)...")

        # 6. Prediction
        logger.info("Skipping Prediction (Not implemented yet)...")

    except Exception as e:
        logger.exception("An error occurred during the pipeline execution.") # Logs the full traceback

    finally:
        end_time = time.time()
        logger.info(f"Pipeline finished in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
