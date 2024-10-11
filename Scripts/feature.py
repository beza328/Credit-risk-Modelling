import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

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



def extract_features(data):
    logger.info("extracting feaure for credit risk model...")

    # 1. Transaction Frequency
    transaction_frequency = data.groupby('CustomerId')['TransactionId'].count().reset_index()
    transaction_frequency.columns = ['CustomerId', 'Transaction_Frequency']

    # 2. Average Transaction Amount
    avg_transaction_amount = data.groupby('CustomerId')['Amount'].mean().reset_index()
    avg_transaction_amount.columns = ['CustomerId', 'Avg_Transaction_Amount']

    # 3. Total Transaction Volume
    total_transaction_volume = data.groupby('CustomerId')['Amount'].sum().reset_index()
    total_transaction_volume.columns = ['CustomerId', 'Total_Transaction_Volume']

    #4. Transaction Timing
    # Calculate Recency
    current_date = pd.to_datetime('now')
    #data['Recency'] = (current_date - data['TransactionStartTime']).dt.days

    # Convert current_date to EAT (UTC+3)
    # Localize current_date to EAT (UTC+3)
    current_date_eat = current_date.tz_localize('Africa/Addis_Ababa')

    # Assuming data['TransactionStartTime'] is in UTC, first localize it to UTC
    #data['TransactionStartTime'] = data['TransactionStartTime'].dt.tz_localize('UTC')

    # Now convert TransactionStartTime to EAT
    data['TransactionStartTime_EAT'] = data['TransactionStartTime'].dt.tz_convert('Africa/Addis_Ababa')

    # Calculate Recency in days
    data['Recency'] = (current_date_eat - data['TransactionStartTime_EAT']).dt.days

    recency = data.groupby('CustomerId')['Recency'].min().reset_index()
    recency.columns = ['CustomerId', 'Transaction_Recency']

    # 5. Variability of Transaction Amounts
    transaction_variability = data.groupby('CustomerId')['Amount'].std().reset_index()
    transaction_variability.columns = ['CustomerId', 'Transaction_Amount_Variability']


    #6. Extracting Time-Based Features from TransactionStartTime
    data['TransactionHour'] = data['TransactionStartTime'].dt.hour
    data['TransactionDay'] = data['TransactionStartTime'].dt.day
    data['TransactionMonth'] = data['TransactionStartTime'].dt.month
    data['TransactionYear'] = data['TransactionStartTime'].dt.year

    # 7. Calculate the mean values for new time-based features for each customer
    transaction_hour = data.groupby('CustomerId')['TransactionHour'].mean().reset_index()
    transaction_hour.columns = ['CustomerId', 'Avg_Transaction_Hour']

    transaction_day = data.groupby('CustomerId')['TransactionDay'].mean().reset_index()
    transaction_day.columns = ['CustomerId', 'Avg_Transaction_Day']

    transaction_month = data.groupby('CustomerId')['TransactionMonth'].mean().reset_index()
    transaction_month.columns = ['CustomerId', 'Avg_Transaction_Month']

    transaction_year = data.groupby('CustomerId')['TransactionYear'].mean().reset_index()
    transaction_year.columns = ['CustomerId', 'Avg_Transaction_Year']

    # Merge all features into a single DataFrame
    features = transaction_frequency.merge(avg_transaction_amount, on='CustomerId') \
                                    .merge(total_transaction_volume, on='CustomerId') \
                                    .merge(recency, on='CustomerId') \
                                    .merge(transaction_variability, on='CustomerId')\
                                    .merge(transaction_hour, on='CustomerId') \
                                    .merge(transaction_day, on='CustomerId') \
                                    .merge(transaction_month, on='CustomerId') \
                                    .merge(transaction_year, on='CustomerId')

    # Reset index if necessary
    features.reset_index(drop=True, inplace=True)
    return features
    # Output the features DataFrame
    print(features.head())
    logger.info("the feature engineering process done.")

def LabelEncoder(features):
    logger.info("encoding catagorical variables...")

    # One-Hot Encoding for nominal categorical variables 
    features = pd.get_dummies(features, columns=['CustomerId'], drop_first=True)
    
    logger.info("Catagorical variables encoded.")


def check_missing_value(features):
    logger.info("checking the missin value...")
    missing_values = features.isnull().sum()
    print(missing_values)
    # Option 1: Imputation
    # Filling missing values with mean, median, or mode for numerical features
    for column in features.select_dtypes(include=['float64', 'int64']).columns:
        features[column].fillna(features[column].mean(), inplace=True) 
    logger.info("Missing value checked")

def scaling(featues):
    logger.info("scaling numerical columns...")
        # Separate numerical features
    numerical_features = features.select_dtypes(include=['float64', 'int64']).columns

    # Option 1: Normalization
    scaler = MinMaxScaler()
    features[numerical_features] = scaler.fit_transform(features[numerical_features])
    logger.info("Numerical columns Scaled.")

