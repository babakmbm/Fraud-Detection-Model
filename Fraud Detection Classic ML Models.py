"""
Fraud Detection Model Training Script

Author: Babak Rahi

This script performs fraud detection model training on a given dataset of transactions.
It includes data preprocessing, feature engineering, and model training with optional 
hyperparameter tuning and MLflow logging. Models are saved for future use.

Usage from the terminal:
    Default: 
        - python Fraud Detection Classic ML Models.py
    Default with requirements installation:
        - python Fraud Detection Classic ML Models.py --install
    Custom file paths and methods:
        - python Fraud Detection Classic ML Models.py --transactions path/to/transactions.csv --labels path/to/labels.csv --data_dict path/to/data-dictionary.xlsx --method account --mlflow --tuning
    
Options:
    --transactions : Path to the transactions CSV file.
    --labels       : Path to the labels CSV file.
    --data_dict    : Path to the data dictionary Excel file.
    --method       : Balancing method to use. Choices are 'account', 'random', or 'smote'. Default is 'account'.
    --tuning       : Perform hyperparameter tuning using RandomizedSearchCV.
    --mlflow       : Enable MLflow logging to track experiments and log models.
    --install      : Install requirements from requirements.txt file.

Suggestions:
    - Use 'account' balancing method for better results.
    - Do not Enable hyperparameter tuning if you are runnign the script on a personal pc.
    - Do not Enable MLflow logging if you are runnign the script on a personal pc and not using cloud resources.
Notes:
    - The script can also be run without any arguments to use the default file paths and best hyperparameters.
    - The default method used for balancing the dataset is an accoount based method of undersampling suitable for real world applications.
    - The default script assumes that the transactions_obf.csv and labels_obf.csv files are in the 'new_data' directory 
      and the data-dictionary.xlsx is in the root directory.

Outputs:
    - Metrics summary saved to results/metrics_summary.csv.
    - Model files saved to the 'models' directory.
    - ROC curve plot saved to the 'plots' directory.
    - MLflow logs saved to the 'mlflow' directory (if MLFlow logging enabled)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from scipy.stats import uniform, randint
import argparse
import os
import joblib
import mlflow
import json
import subprocess
import sys

# This function is used to load the data from the CSV and Excel files
# this is called in the main function
def load_data(transactions_path, labels_path, data_dict_path):
    labels_df = pd.read_csv(labels_path)
    transactions_df = pd.read_csv(transactions_path)
    data_dict = pd.read_excel(data_dict_path)
    return transactions_df, labels_df, data_dict

# This function is used to preprocess the data in the transactions and labels dataframes and create a merged dataframe
# this is called in the main function
def preprocess_data(transactions_df, labels_df):
    transactions_df['transactionTime'] = pd.to_datetime(transactions_df['transactionTime'])
    transactions_df['merchantZip'].fillna('Unknown', inplace=True)
    # the posEntryMode column is  considered as categorical column and needs to be encoded later in the code
    transactions_df['posEntryMode'] = transactions_df['posEntryMode'].astype('category').cat.codes
    merged_df = pd.merge(transactions_df, labels_df, on='eventId', how='left')
    merged_df['fraud_flag'] = merged_df['reportedTime'].notnull().astype(int)
    # droping the reportedTime column as it is not needed for real-time fraud detection
    merged_df.drop(columns=['reportedTime'], inplace=True)
    return merged_df

"""
To transform the raw transactional data into a set of meaningful features 
that can be used to train machine learning models for fraud detection.
"""
def feature_engineering(merged_df):
    # Double check that the transactionTime field is in a consistent datetime format for further analysis.
    # Enables the extraction of time-based features that are crucial for detecting patterns indicative of fraud.
    merged_df['transactionTime'] = pd.to_datetime(merged_df['transactionTime'])
    # Helps in identifying fraudulent activities that often occur at unusual hours or specific days of the week when legitimate transactions are less frequent
    merged_df['hour'] = merged_df['transactionTime'].dt.hour
    merged_df['day_of_week'] = merged_df['transactionTime'].dt.dayofweek
    merged_df['month'] = merged_df['transactionTime'].dt.month
    # Detects anomalies such as an unusually high number of transactions in a short period, which is a common indicator of fraud
    merged_df['transaction_date'] = merged_df['transactionTime'].dt.date
    merged_df['transactions_per_day'] = merged_df.groupby(['accountNumber', 'transaction_date'])['eventId'].transform('count')
    # Identifies deviations from normal spending patterns, which can signal fraudulent behavior
    merged_df['mean_transaction_amount_by_account'] = merged_df.groupby('accountNumber')['transactionAmount'].transform('mean')
    merged_df['std_transaction_amount_by_account'] = merged_df.groupby('accountNumber')['transactionAmount'].transform('std').fillna(0)
    # Detects rapid successive transactions, which are often associated with compromised accounts
    merged_df['time_since_last_transaction'] = merged_df.sort_values(by=['accountNumber', 'transactionTime']).groupby('accountNumber')['transactionTime'].diff().dt.total_seconds().fillna(0)
    # Helps in identifying transactions that deplete an account unusually fast
    merged_df['transaction_to_cash_ratio'] = merged_df['transactionAmount'] / merged_df['availableCash']
    # Keeps a record of the last transaction amount for each account. 
    # Detects significant deviations from typical transaction amounts
    merged_df['previous_transaction_amount'] = merged_df.sort_values(by=['accountNumber', 'transactionTime']).groupby('accountNumber')['transactionAmount'].shift().fillna(0)
    # Monitors overall spending behavior and flags significant changes that may be fraudulent.
    merged_df['cumulative_sum_by_account'] = merged_df.groupby('accountNumber')['transactionAmount'].cumsum()
    merged_df['cumulative_count_by_account'] = merged_df.groupby('accountNumber').cumcount() + 1

    """
    Handeling Outliers:
    Observations from Outlier Analysis
        - Transaction Amounts: A significant number of outliers, with many transactions exceeding the upper quartile limit by a large margin. 
        - Approximately 1.6% of the transactions identified as outliers in transaction amounts are fraudulent. This is a higher proportion than in the overall dataset.
        - Available Cash: The availableCash also shows some outliers, but these are fewer and less extreme compared to transaction amounts. 
        - Only about 0.2% of the transactions identified as outliers in available cash are fraudulent. This is a very small proportion.
    Method to Handle outliers 
        - we can check the distribution of fraud within the outliers using the fraud_flag (Check the Proportion of Fraud in Outliers). 
        - This will help us understand if the outliers are more likely to be fraudulent, which can inform our decision on how to handle them.
        - Given these observations, it is essential to retain the transaction amount outliers in the dataset as they have a higher likelihood of being fraudulent. 
        - Instead of removing or capping outliers arbitrarily, we Create a feature that flags extreme values, which might indicate fraud and Apply log transformation to handle skewness while keeping extreme values meaningful
    """

    Q1 = merged_df['transactionAmount'].quantile(0.25)
    Q3 = merged_df['transactionAmount'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold_high = Q3 + 1.5 * IQR
    merged_df['is_outlier_transactionAmount'] = (merged_df['transactionAmount'] > outlier_threshold_high).astype(int)

    Q1 = merged_df['availableCash'].quantile(0.25)
    Q3 = merged_df['availableCash'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold_high = Q3 + 1.5 * IQR
    merged_df['is_outlier_availableCash'] = (merged_df['availableCash'] > outlier_threshold_high).astype(int)

    # Apply log transformation to handle skewness while keeping extreme values meaningful 
    merged_df['log_transactionAmount'] = np.log1p(merged_df['transactionAmount'])
    merged_df['log_availableCash'] = np.log1p(merged_df['availableCash'])

    merged_df.drop(columns=['transaction_date'], inplace=True)

    # We convert the categorical variables to numerical format using label encoding to prepare the data for machine learning models
    label_encoder = LabelEncoder()
    for col in ['accountNumber', 'merchantId', 'merchantZip']:
        merged_df[col] = label_encoder.fit_transform(merged_df[col])
    merged_df['posEntryMode'] = merged_df['posEntryMode'].astype('int')
    merged_df['merchantCountry'] = merged_df['merchantCountry'].astype('int')
    merged_df = merged_df.drop(columns=['transactionTime', 'eventId'])

    # We Converts categorical variables to numerical format using one-hot encoding 
    # to avoide the ordinal nature implied by label encoding and represent each category with a binary vector
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(merged_df[['posEntryMode', 'merchantCountry']])
    encoded_feature_names = encoder.get_feature_names_out(['posEntryMode', 'merchantCountry'])
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    merged_df = pd.concat([merged_df.drop(columns=['posEntryMode', 'merchantCountry']), encoded_df], axis=1)

    """
    By performing feature scaling, we kept all numeric explanatory variables within the same domain 
    by using range transformation to compute all numeric variables to be in a range of 0 and 1
    This is crucial for the performance of machine learning algorithms
    """
    scaler = MinMaxScaler()
    numeric_features = ['transactionAmount', 'availableCash', 'hour', 'day_of_week', 'month', 'transactions_per_day', 
                        'mean_transaction_amount_by_account', 'std_transaction_amount_by_account', 'time_since_last_transaction', 
                        'transaction_to_cash_ratio', 'previous_transaction_amount', 'cumulative_sum_by_account', 'cumulative_count_by_account']
    merged_df[numeric_features] = scaler.fit_transform(merged_df[numeric_features])

    return merged_df

"""
    To address class imbalance in the dataset we can used undersampling, oversampling, or a combination of both
    in a real world scenario oversampling using SMOTE could be problematic as it can generate synthetic samples 
    that are too similar to existing ones and lead to overfitting
    
    -- Account Based Undersampling: 
       We perform undersampling based on accounts rather than random undersampling. 
       We balanced the data by reducing the number of transactions from accounts with non-fraudulent transactions 
       to match the number of transactions from accounts with fraudulent transactions. 
       This approach ensures that the distribution of transactions per account remains realistic 
       and avoids the potential bias introduced by random undersampling.

    Note: code below allows for comparison of different balancing methods
"""

def balance_data(merged_df, method='account'):
    features = merged_df.drop(columns=['fraud_flag'])
    target = merged_df['fraud_flag']

    # Main method used in our generated model
    if method == 'account':
        account_balanced_data = []
        for account in merged_df[merged_df['fraud_flag'] == 1]['accountNumber'].unique():
            account_data = merged_df[merged_df['accountNumber'] == account]
            fraud_transactions = account_data[account_data['fraud_flag'] == 1]
            non_fraud_transactions = account_data[account_data['fraud_flag'] == 0]
            num_fraud_transactions = len(fraud_transactions)
            if len(non_fraud_transactions) < num_fraud_transactions:
                non_fraud_sample = non_fraud_transactions
            else:
                non_fraud_sample = non_fraud_transactions.sample(n=num_fraud_transactions, random_state=42)
            account_balanced_data.append(pd.concat([fraud_transactions, non_fraud_sample]))
        balanced_df = pd.concat(account_balanced_data)
    # Random undersampling method for comparison - to run this method give the flag --method random when running script from the terminal
    elif method == 'random':
        undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = undersample.fit_resample(features, target)
        balanced_df = pd.concat([pd.DataFrame(X_resampled, columns=features.columns), pd.Series(y_resampled, name='fraud_flag')], axis=1)
    # SMOTE oversampling method for comparison - to run this method give the flag --method smote when running script from the terminal
    elif method == 'smote':
        smote = SMOTE(random_state=42)
        X_resampled_smote, y_resampled_smote = smote.fit_resample(features, target)
        balanced_df = pd.concat([pd.DataFrame(X_resampled_smote, columns=features.columns), pd.Series(y_resampled_smote, name='fraud_flag')], axis=1)
    else:
        raise ValueError("Invalid balancing method. Choose 'account', 'random', or 'smote'.")

    return balanced_df

# This section is again optional for hyperparameter tuning and MLflow logging
param_dist_lr = {
    'C': uniform(0.1, 10),
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': randint(100, 500)
}

param_dist_rf = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(10, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4),
    'bootstrap': [True, False]
}

param_dist_dt = {
    'max_depth': randint(10, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4),
    'criterion': ['gini', 'entropy']
}

param_dist_xgb = {
    'n_estimators': randint(100, 300),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(10, 30),
    'subsample': uniform(0.8, 0.2),
    'colsample_bytree': uniform(0.8, 0.2)
}

# Function to perform randomized search - has been used in the main function as optional
def perform_random_search(model, param_dist, X_train, y_train):
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)
    print(f"Best parameters for {model.__class__.__name__}:", random_search.best_params_)
    return random_search.best_estimator_

# Function to evaluate models and plot ROC curves
def train_evaluate_and_plot_roc(X_train, X_test, y_train, y_test, dataset_name, hyperparameter_tuning, mlflow_logging):

    # Hyperparameters for models as the result of hyperparameter tuning - This section runs by default
    default_models = {
        'Logistic Regression': LogisticRegression(C=9.756320330745593, max_iter=341, solver='liblinear'),
        'Random Forest': RandomForestClassifier(n_estimators=250, max_depth=20, min_samples_split=5, min_samples_leaf=2, bootstrap=True),
        'Decision Tree': DecisionTreeClassifier(max_depth=25, min_samples_split=5, min_samples_leaf=2, criterion='gini'),
        'XGBoost': XGBClassifier(n_estimators=250, learning_rate=0.15, max_depth=20, subsample=0.9, colsample_bytree=0.9)
    }
    
    # models to be used as a starting point for hyperparameter tuning - This section runs if hyperparameter_tuning is enabled
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    plt.figure(figsize=(12, 8))
    # Loop through the models and train them on the training data
    for model_name, model in models.items():
        if hyperparameter_tuning:
            best_model = perform_random_search(model, param_dist_lr if model_name == 'Logistic Regression' else
                                               param_dist_rf if model_name == 'Random Forest' else
                                               param_dist_dt if model_name == 'Decision Tree' else
                                               param_dist_xgb, X_train, y_train)
        else: # This runs if hyperparameter tuning is not enabled - runs with already tuned hyperparameters
            best_model = default_models[model_name]
            best_model.fit(X_train, y_train)
        
        # This runs if MLflow logging is enabled
        if mlflow_logging:
            with mlflow.start_run(run_name=f"{model_name} - {dataset_name}"):
                mlflow.log_params(best_model.get_params())
                mlflow.sklearn.log_model(best_model, model_name)
                
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]

                # Calculate evaluation metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                report = classification_report(y_test, y_pred, output_dict=True)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                results[model_name] = {
                    "classification_report": report,
                    "roc_auc": roc_auc,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "confusion_matrix": cm
                }
                
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
                
                mlflow.log_metrics({
                    "roc_auc": roc_auc,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                })
                mlflow.log_dict(report, f"{model_name}_classification_report.json")
                mlflow.end_run()
        else:
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            report = classification_report(y_test, y_pred, output_dict=True)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[model_name] = {
                "classification_report": report,
                "roc_auc": roc_auc,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm
            }
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        # Save the models in a folder named 'models'
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, f'models/{model_name}_{dataset_name.replace(" ", "_")}.joblib')
        
        # In case we want the models saved as pickle files uncomment this
        # joblib.dump(best_model, f"{model_name.replace(' ', '_')}_{dataset_name.replace(' ', '_')}_model.pkl")
    
    # ROC curve plot settings and saving
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend()
    # save the plot in a folder named 'plots'
    os.makedirs('plots', exist_ok=True)
    plot_filename = f'plots/{dataset_name.replace(" ", "_")}_roc_curve.png'
    plt.savefig(plot_filename)

    # In case we are using MLflow, log the plot as an artifact in the run
    if mlflow_logging:
        mlflow.log_artifact(plot_filename)
    
    return results


# Main function that calls all the other functions
def main(transactions_path='data-new/transactions_obf.csv', labels_path='data-new/labels_obf.csv', data_dict_path='data-dictionary.xlsx', method='account', hyperparameter_tuning=False, mlflow_logging=False):
    transactions_df, labels_df, data_dict = load_data(transactions_path, labels_path, data_dict_path)
    merged_df = preprocess_data(transactions_df, labels_df)
    merged_df = feature_engineering(merged_df)
    balanced_df = balance_data(merged_df, method=method)

    features = balanced_df.drop(columns=['fraud_flag'])
    target = balanced_df['fraud_flag']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    results = train_evaluate_and_plot_roc(X_train, X_test, y_train, y_test, f"{method.capitalize()} Balancing", hyperparameter_tuning, mlflow_logging)
    # Print and save results for inspection
    metrics_list = []
    for model_name, metrics in results.items():
        print(f"Model: {model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        print("Classification Report:\n", pd.DataFrame(metrics['classification_report']).transpose())
        
        metrics_entry = {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC AUC': metrics['roc_auc'],
            'Confusion Matrix': json.dumps(metrics['confusion_matrix'].tolist()),  # Convert to list for JSON compatibility
            'Classification Report': json.dumps(metrics['classification_report'])
        }
        metrics_list.append(metrics_entry)
    
    metrics_df = pd.DataFrame(metrics_list)
    # create a results folder to save the metrics summary if it does not exist
    os.makedirs('results', exist_ok=True)
    metrics_df.to_csv('results/metrics_summary.csv', index=False)
    print("Metrics saved to results/metrics_summary.csv")

# Function to install requirements
def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# This section is used to run the main function from the terminal if the script is run directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fraud Detection Model Training')
    # flags for terminal arguments
    parser.add_argument('--transactions', type=str, help='Path to the transactions CSV file')
    parser.add_argument('--labels', type=str, help='Path to the labels CSV file')
    parser.add_argument('--data_dict', type=str, help='Path to the data dictionary Excel file')
    parser.add_argument('--method', type=str, choices=['account', 'random', 'smote'], default='account', help='Balancing method to use')
    parser.add_argument('--tuning', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow logging')
    parser.add_argument('--install', action='store_true', help='Install requirements')
    args = parser.parse_args()

    if args.install:
        install_requirements()

    if args.transactions and args.labels and args.data_dict:
        main(args.transactions, args.labels, args.data_dict, args.method, args.tuning, args.mlflow)
    else:
        main()
