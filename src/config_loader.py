import yaml
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine the absolute path to the project root directory
# This assumes config_loader.py is in the 'src' directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Loads the YAML configuration file.

    Args:
        config_path (str): The path to the configuration file.
                           Defaults to 'config/config.yaml' in the project root.

    Returns:
        dict: A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the configuration file.
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded successfully from: {config_path}")
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path}: {e}")
        raise yaml.YAMLError(f"Error parsing configuration file {config_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading config {config_path}: {e}")
        raise

if __name__ == '__main__':
    # Example usage: Load the default config and print it
    try:
        config = load_config()
        print("Configuration loaded:")
        import json
        print(json.dumps(config, indent=2))

        # Accessing a specific value
        data_path = config.get('data', {}).get('raw_data_path')
        if data_path:
            print(f"\nRaw data path from config: {data_path}")
        else:
            print("\nRaw data path not found in config.")

    except (FileNotFoundError, yaml.YAMLError, Exception) as e:
        print(f"Failed to load configuration: {e}")
