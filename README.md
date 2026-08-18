

# Customer Chur Perdiction with Logistic Regression Model 

## Overview

The following ML model perdicts customer churn using a Logistic Regression Model. 
The model is trained and evaluated using the Telco Customer Churn dataset. 
The model is optimized using 5-fold cross-validation and hyperparamter tuning using GridSearchCV

## Requirements 

- Python 3.14+
- pip , python package manager 

# Python Packages needed include: 

- pandas 
- skikit-learn


## Hardward 

No GPU is required to run the program. The program uses standard CPU with atleast 4 GB RAM

## Data 

- Dataset `data/CustomerChurnData.csv`

## Overall Project Struture 

d683-advanced-ai-and-ml/
├── data/
│ └── CustomerChurnData.csv
├── src/
│ ├── preprocessing.py
│ └── models/
│ └── logistic_regression.py
└── README.md


## Instructions on how to Run Program : 

1. Start by cloning the repository and navigage to the project root:

```bash 
git clone <https://gitlab.com/wgu-gitlab-environment/student-repos/mha2360/d683-advanced-ai-and-ml.git> 

cd d683-advanced-ai-and-ml
```

2. Create and activate a virtual environment 

```bash

python -m venv .venv
source .venv/bin/activate 



```
#For Windows PC : .venv\Scripts\activate

3. Install the needed Dependencies: 

```bash

pip install pandas skikit-learn 


```

5. TIME TO RUN THE PROGRAM woohoo!

```bash

python -m src.models.logistic_regression



```

## What the script does 

- Loads in data and completes preprocessing of dataset (`src/preprocessing.py`)
- The data is then split into 80% for training and 20% for testing. 
- A GridSearchCV hyperparameters search over the following : `C`, `class_weight`, and `solver`. This is done using 5-fold cross-validation scored on F1

- The following information prints : 
    - Preprocessing Results 
    - Best hyperparameters
    - Cross-validation performance including accuracy, recall, F1, ROC-AUC, mean and standard deviation across folds
    - Evaluated the final tuned model and prints accuracy, recall, F1, and ROC-AUC


## Models Overall Performance: 

Metric | Score 

Accuracy : 74.9%
Recall : 82.57%
F1 : 63.57%
ROC-AUC : 86.21%


**Note:** `class_weight="balanced"` was selected during tuning to prioritize recall, since the dataset is imbalanced (26.5% churn rate) and failing to identify a customer who will churn is more costly than a false positive in a business context. Also Gradient Boosting and Random Forest were also implemented and evaluated as candidate models; Logistic Regression was selected as the final model based on its stronger performance across accuracy, recall, F1, and ROC-AUC during initial comparison.




