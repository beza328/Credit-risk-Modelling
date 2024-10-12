import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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


def logistic_regression_model_evaluation(features):
   # Prepare data for logistic regression model evaluation
    logger.info("logistic regresion Model_evaluation.....")
    # Use only WoE transformed features and Risk_Label for modeling
    X = features[['Recency_WoE', 'Transaction_Frequency', 'Total_Transaction_Volume']]
    y = features['Risk_Label'].map({'Good': 1, 'Bad': 0})  # Encode labels as 1 for Good, 0 for Bad

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    # Make predictions
    y_pred = logistic_model.predict(X_test)

    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)   
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, logistic_model.predict_proba(X_test)[:, 1])
    f1 = f1_score(y_test, y_pred)


    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Displaying the evaluation results
    # Printing the evaluation metrics

    print("Accuracy: {:.2f}%".format(accuracy * 100))

    print("Precision: {:.2f}".format(precision))

    print("Recall: {:.2f}".format(recall))

    print("F1 Score: {:.2f}".format(f1))

    print("ROC-AUC: {:.2f}".format(roc_auc))

    print("Confusion Matrix:")

    print(conf_matrix)
        
def random_forest_model_evaluation(features):
    logger.info("random forest Model_evaluation.....")


    X = features[['Recency_WoE', 'Transaction_Frequency', 'Total_Transaction_Volume']]
    y = features['Risk_Label'].map({'Good': 1, 'Bad': 0})  # Encode labels as 1 for Good, 0 for Bad


    # Splitting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fitting the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Making predictions
    y_pred = rf_model.predict(X_test)

    # Model Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ROC-AUC Score
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print the results
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc,
        'Confusion Matrix': conf_matrix
    }

    for metric, value in results.items():
        print(f"{metric}: {value}")


def fitting_model(X, y):
    logger.info("fitting logistic regression model...")
    # Fit the final model on the entire dataset
    final_model = LogisticRegression(random_state=42)
    final_model.fit(X, y)