import pandas as pd
import logging
import sys
import os

# Step 1: Define the path to the logs directory
log_dir = os.path.join(os.getcwd(), 'logs')  # Use current working directory

# Create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define file paths
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

# Step 2: Create handlers
info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

# Step 3: Create a stream handler to output logs to the notebook
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)  # Output all logs to the notebook

# Step 4: Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Step 5: Create a logger and set its level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all logs (DEBUG and above)
logger.addHandler(info_handler)
logger.addHandler(error_handler)
logger.addHandler(stream_handler)  # Add stream handler for notebook output


# Module for loading CSV data
def load_data(file_path):
    try:
        logger.info(f"Attempting to load data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"No data in file: {file_path}")
        return None
    except pd.errors.ParserError:
        logger.error(f"Error parsing file: {file_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None

def clean_data(data):
    logger.info("Starting data cleaning process...")
    
    # Log initial data info
    logger.info(f"Initial data shape: {data.shape}")
    
    # Handling missing values
    logger.info("Checking for missing values...")
    missing_values = data.isnull().sum()
    logger.info(f"Missing values in each column:\n{missing_values[missing_values > 0]}")
    
    # Fill missing values with mean (for numeric columns)
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        if data[col].isnull().any():
            mean_value = data[col].mean()
            data[col].fillna(mean_value, inplace=True)
            logger.info(f"Filled missing values in '{col}' with mean: {mean_value}")

    # Removing duplicates
    logger.info("Checking for duplicates...")
    initial_shape = data.shape
    data.drop_duplicates(inplace=True)
    logger.info(f"Removed duplicates: {initial_shape[0] - data.shape[0]} rows removed.")
    return data



def data_overview(data):
    logger.info("Checking categorical values...")
    
    for col in data.select_dtypes(include=['object', 'category']).columns:
        unique_values = data[col].unique()
        logger.info(f"Unique values in '{col}': {unique_values}")

        # Count the frequency of each unique value
        value_counts = data[col].value_counts()
        logger.info(f"Value counts for '{col}':\n{value_counts}")
