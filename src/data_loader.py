import pandas as pd
import os
import logging
from typing import Optional

from src.config_loader import load_config, PROJECT_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(config_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Loads the raw data from the JSON file specified in the configuration.

    Args:
        config_path (Optional[str]): Path to the configuration file.
                                     If None, uses the default path.

    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame containing the loaded data,
                                or None if loading fails.
    """
    try:
        # Load configuration
        if config_path:
            config = load_config(config_path)
        else:
            config = load_config() # Uses default path

        # Get the raw data path from config
        raw_data_path_relative = config.get('data', {}).get('raw_data_path')
        if not raw_data_path_relative:
            logging.error("Raw data path ('data.raw_data_path') not found in configuration.")
            return None

        # Construct the absolute path if the configured path is relative
        # If it's already absolute (like the default C:/...), use it directly
        if os.path.isabs(raw_data_path_relative):
            raw_data_path_absolute = raw_data_path_relative
        else:
            raw_data_path_absolute = os.path.join(PROJECT_ROOT, raw_data_path_relative)

        logging.info(f"Attempting to load data from: {raw_data_path_absolute}")

        # Check if the file exists
        if not os.path.exists(raw_data_path_absolute):
            logging.error(f"Data file not found at: {raw_data_path_absolute}")
            raise FileNotFoundError(f"Data file not found at: {raw_data_path_absolute}")

        # Load the JSON data
        # Try standard JSON loading first
        try:
            df = pd.read_json(raw_data_path_absolute)
            logging.info(f"Successfully loaded data from {raw_data_path_absolute}. Shape: {df.shape}")
        except ValueError as json_error:
            logging.warning(f"Standard JSON loading failed: {json_error}. Trying JSON Lines format (lines=True).")
            try:
                df = pd.read_json(raw_data_path_absolute, lines=True)
                logging.info(f"Successfully loaded data as JSON Lines from {raw_data_path_absolute}. Shape: {df.shape}")
            except Exception as lines_error:
                logging.error(f"Failed to load data as standard JSON or JSON Lines from {raw_data_path_absolute}: {lines_error}")
                raise lines_error

        # Flatten the 'items' list so each item becomes its own row
        if 'items' in df.columns:
            records = []
            for _, row in df.iterrows():
                parent_data = row.drop('items').to_dict()
                for item in row['items']:
                    rec = parent_data.copy()
                    rec.update(item)
                    records.append(rec)
            df = pd.DataFrame(records)
            logging.info(f"Flattened items. New shape: {df.shape}")
        else:
            logging.warning("No 'items' column found to flatten.")
        return df

    except FileNotFoundError as fnf_error:
        # Already logged in the checks above or by load_config
        logging.error(f"File not found during data loading: {fnf_error}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        return None

if __name__ == '__main__':
    # Example usage: Load the data using the default config and print info
    print("Attempting to load data...")
    data_df = load_data()

    if data_df is not None:
        print("\nData loaded successfully!")
        print("\nFirst 5 rows:")
        print(data_df.head())
        print("\nData Info:")
        data_df.info()
    else:
        print("\nFailed to load data. Check logs for details.")
