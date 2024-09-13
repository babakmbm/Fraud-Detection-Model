
# Fraud Detection Model Training

This repository contains a Python-based fraud detection model training script. It uses a dataset of transactional data, applies data preprocessing and feature engineering, and trains machine learning models to detect fraudulent transactions. The script offers multiple balancing methods and can optionally perform hyperparameter tuning and MLflow logging.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Balancing Methods](#balancing-methods)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Results](#results)
8. [License](#license)

---

## Overview
The aim of this project is to build an efficient fraud detection model that identifies fraudulent transactions from a given dataset. We use techniques like undersampling, SMOTE, and account-based sampling to handle imbalanced datasets.

The script can:
- Load transaction and label datasets.
- Preprocess and engineer relevant features.
- Apply data balancing methods.
- Train and evaluate multiple machine learning models.
- Save metrics, models, and plots for further analysis.

---

## Features
- **Data Preprocessing:** Clean and transform raw transactional data.
- **Feature Engineering:** Create new features that are likely indicators of fraud, including time-based features and outlier detection.
- **Balancing Methods:** Perform undersampling, oversampling, or a hybrid approach to deal with data imbalance.
- **Hyperparameter Tuning (Optional):** Tune hyperparameters using RandomizedSearchCV.
- **MLflow Logging (Optional):** Log experiments using MLflow for model tracking.
- **Metrics Summary:** Save and display evaluation metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **ROC Curve:** Visualize the model performance.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/fraud_detection.git
cd fraud_detection
```

### Install Dependencies
You can install the required Python packages by running:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Default Script
To run the default script with provided datasets:
```bash
python Babak_Rahi_FeatureSpace.py
```

### Customize File Paths
To specify custom paths for the dataset:
```bash
python Babak_Rahi_FeatureSpace.py --transactions path/to/transactions.csv --labels path/to/labels.csv --data_dict path/to/data-dictionary.xlsx --method account --mlflow --tuning
```

### Options:
- **`--transactions`** : Path to the transactions CSV file.
- **`--labels`** : Path to the labels CSV file.
- **`--data_dict`** : Path to the data dictionary Excel file.
- **`--method`** : Choose a balancing method (`account`, `random`, `smote`). Default is `account`.
- **`--tuning`** : Perform hyperparameter tuning (optional).
- **`--mlflow`** : Enable MLflow logging (optional).
- **`--install`** : Install required packages (optional).

---

## Balancing Methods

We experimented with three data balancing techniques: 
1. **Account-Based Undersampling:** Sample based on account transactions to preserve data distribution.
2. **Random Undersampling:** Reduce the number of non-fraudulent transactions randomly.
3. **SMOTE Oversampling:** Generate synthetic samples for the minority class (fraudulent transactions).

### Example: ROC Curves for Balancing Methods
**Random Undersampling**:
![ROC Random Undersampling](plots/EDAPlots/ROC-Random_Undersampling.png)

**Account-Based Undersampling**:
![Fraud vs Non-Fraud Transaction Count (Account)](plots/EDAPlots/Fraud_vs_Non-Fraud_Transaction_Count_Account-Level_Undersampling.png)

---

## Model Training and Evaluation

The script trains four machine learning models:
- Logistic Regression
- Random Forest
- Decision Tree
- XGBoost

Each model is evaluated based on multiple metrics, including Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Hyperparameter tuning is available if needed.

### Example: Correlation Matrix of Features
![Correlation Matrix](plots/EDAPlots/Correlation_Matrix.png)

### Example: Boxplots of Features
**Boxplot of Available Cash:**
![Boxplot Available Cash](plots/EDAPlots/Boxplot_of_Available_Cash.png)

**Boxplot of Transaction Amounts:**
![Boxplot Transaction Amounts](plots/EDAPlots/Boxplot_of_Transaction_Amounts.png)

### Example: Distribution of Fraudulent Transactions
**Distribution by Fraud Status:**
![Distribution of Available Cash by Fraud Status](plots/EDAPlots/Distribution_of_Available_Cash_by_Fraud_Status.png)

---

## Results

### Model Comparison: ROC Curves
The ROC curve below shows the model performance for different balancing techniques.

![ROC Random Undersampling](plots/EDAPlots/ROC-Random_Undersampling.png)

### Fraud Detection Metrics
After running the models, the results are saved in the `results/metrics_summary.csv` file.

---

## License

This project is licensed under the MIT License.
