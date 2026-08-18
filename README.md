

# Customer Churn Prediction with Machine Learning

## Overview

This project uses supervised machine learning to predict customer churn for a subscription-based telecommunications company. The goal is to identify customers who are at risk of discontinuing their services and provide data-driven insights that can support customer retention strategies.

The project uses the Telco Customer Churn dataset and evaluates multiple classification algorithms, including Logistic Regression, Random Forest, and Gradient Boosting.

After comparing the models, Logistic Regression was selected as the final model based on its performance across multiple evaluation metrics.

## Project Objectives

The primary objectives of this project were to:

* Predict whether a customer is likely to churn.
* Compare multiple supervised machine learning classification models.
* Address class imbalance within the dataset.
* Optimize model performance using hyperparameter tuning.
* Evaluate the model using multiple classification metrics.
* Use explainable AI techniques to better understand churn predictions.

## Dataset

The project uses the Telco Customer Churn dataset from Kaggle.

The dataset contains 7,043 customer records and includes information about customer demographics, account information, services, billing, and payment methods.

The target variable is `Churn`, which indicates whether a customer discontinued their service.

Dataset source:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Machine Learning Models

Three classification algorithms were evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting

Logistic Regression was selected as the final model based on its overall performance during model comparison.

## Data Preprocessing

The preprocessing workflow includes:

* Handling missing values
* Encoding categorical variables
* Preparing numerical features
* Splitting the dataset into training and testing sets
* Addressing class imbalance

The dataset was divided into 80% training data and 20% testing data.

## Model Optimization

The Logistic Regression model was optimized using GridSearchCV with 5-fold cross-validation.

The hyperparameters evaluated included:

* `C`
* `class_weight`
* `solver`

The model was optimized using F1 score during the grid search.

Because the dataset contains an imbalanced target variable, `class_weight="balanced"` was selected. This prioritizes identifying customers who are likely to churn, which is important in a customer retention use case.

## Model Performance

The final tuned Logistic Regression model achieved the following results on the test dataset:

| Metric   |  Score |
| -------- | -----: |
| Accuracy | 74.90% |
| Recall   | 82.57% |
| F1 Score | 63.57% |
| ROC-AUC  | 86.21% |

The model achieved an ROC-AUC score above the project's target of 85%.

## Explainable AI

SHAP, or SHapley Additive Explanations, was incorporated to provide insight into the features influencing customer churn predictions.

Explainability is important because a prediction alone does not explain why a customer may be considered high risk. Understanding the factors behind predictions can help organizations make more informed customer retention decisions.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* SHAP
* GridSearchCV
* Visual Studio Code
* Git/GitHub

## Project Structure

```text
d683-advanced-ai-and-ml/
│
├── data/
│   └── CustomerChurnData.csv
│
├── src/
│   ├── preprocessing.py
│   └── models/
│       └── logistic_regression.py
│
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone (https://github.com/MarteniqueH/customer-churn-logistic-regression.git)
cd d683-advanced-ai-and-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment:

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas scikit-learn
```

If SHAP is included in the implementation:

```bash
pip install shap
```

### 4. Run the model

```bash
python -m src.models.logistic_regression
```

## What the Program Does

When the program runs, it:

1. Loads the customer churn dataset.
2. Preprocesses the customer data.
3. Splits the data into training and testing sets.
4. Performs a GridSearchCV hyperparameter search.
5. Uses 5-fold cross-validation to evaluate model performance.
6. Trains the optimized Logistic Regression model.
7. Evaluates the final model using accuracy, recall, F1 score, and ROC-AUC.
8. Displays the model's performance results.

## Hardware

This project does not require a GPU and can run using a standard CPU with at least 4 GB of RAM.

The project was developed on an Apple Mac mini with an M1 chip and 8 GB of memory running macOS.

## Future Improvements

Potential future improvements include:

* Building an interactive web application for customer churn predictions.
* Adding additional machine learning algorithms for comparison.
* Expanding SHAP-based model explanations.
* Implementing automated model retraining.
* Adding customer-level churn probability predictions.
* Developing a dashboard for business users to monitor churn risk.

