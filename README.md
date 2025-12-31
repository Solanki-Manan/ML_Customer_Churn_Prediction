# Bank Customer Churn Prediction

This project analyzes bank customer data to predict customer churn using various machine learning models. The analysis includes data preprocessing, exploratory data analysis, and model evaluation.

## Dataset

The dataset contains information about bank customers, including:
*   **Customer ID:** Unique identifier for each customer.
*   **Credit Score:** Credit score of the customer.
*   **Country:** Country of residence (France, Spain, Germany).
*   **Gender:** Gender of the customer.
*   **Age:** Age of the customer.
*   **Tenure:** Number of years the customer has been with the bank.
*   **Balance:** Account balance.
*   **Products Number:** Number of bank products the customer uses.
*   **Credit Card:** Whether the customer has a credit card (1=Yes, 0=No).
*   **Active Member:** Whether the customer is an active member (1=Yes, 0=No).
*   **Estimated Salary:** Estimated annual salary.
*   **Churn:** Target variable (1=Churned, 0=Not Churned).

## Preprocessing Steps

The following preprocessing steps were applied to the data:
1.  **Categorical Encoding:**
    *   `Gender`: Label Encoded (Female=0, Male=1).
    *   `Country`: One-Hot Encoded (France, Germany, Spain).
2.  **Feature selection:** Dropped `customer_id` and `Country` (original column) after encoding.
3.  **Feature Scaling:** `StandardScaler` was used to scale numerical features: `credit_score`, `age`, `tenure`, `balance`, `products_number`, and `estimated_salary`.
4.  **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique) was used to balance the dataset, as the original dataset had a class imbalance (approx. 80% non-churn vs. 20% churn).

## Models Implemented

The following machine learning models were trained and evaluated:
1.  **Random Forest Classifier**
2.  **K-Nearest Neighbors (KNN)** (n_neighbors=3)
3.  **Gaussian Naive Bayes (GNB)**
4.  **Artificial Neural Network (ANN)**
    *   Architecture: Inputs -> Dense(3, relu) -> Dense(4, relu) -> Dense(1, sigmoid)
    *   Optimizer: Adam
    *   Loss: Binary Crossentropy

## Evaluation Metrics

The models were evaluated using the following metrics:
*   **Accuracy**
*   **Precision**
*   **Recall**
*   **F1 Score**

## Results Summary

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | ~90.3% | ~92.2% | ~87.6% | ~89.8% |
| **KNN** | ~86.0% | ~85.5% | ~85.9% | ~85.7% |
| **Gaussian NB** | ~82.1% | ~81.9% | ~81.3% | ~81.6% |
| **Neural Network** | (Varies per run) | (Varies) | (Varies) | (Varies) |

*Note: The Neural Network metrics shown in the notebook vary slightly with each training run.*

## Requirements

To run this notebook, you need the following Python libraries:
*   pandas
*   numpy
*   matplotlib
*   scikit-learn
*   tensorflow (or keras)
*   imblearn

## Usage

1.  Ensure the dataset `churn.csv` is in the correct path or update the loading path in the notebook.
2.  Run the cells in `Final.ipynb` sequentially to perform preprocessing, training, and evaluation.
