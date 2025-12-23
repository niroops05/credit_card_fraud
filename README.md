CREDIT CARD FRAUD DETECTION DASHBOARD
===================================

PROJECT OVERVIEW
----------------
This project implements an end-to-end Credit Card Fraud Detection system
using Machine Learning and deploys it as an interactive Streamlit dashboard.

The system is designed to help financial institutions identify fraudulent
transactions while balancing fraud risk and customer experience.

The project uses an XGBoost model with RFECV-based feature selection and
selective feature scaling, ensuring consistency between training and deployment.


KEY FEATURES
------------
- XGBoost-based fraud detection model
- RFECV for feature selection
- Selective scaling of transaction amount
- Interactive Streamlit dashboard
- Dynamic fraud probability threshold tuning
- Interactive confusion matrix
- Feature importance visualization
- Manual fraud prediction using:
  - Top 5 most important features
  - Transaction Amount always included
  - Remaining features auto-filled using dataset medians


PROJECT STRUCTURE
-----------------
credit_card/
|
|-- app.py                 -> Streamlit dashboard
|-- main.ipynb             -> Model training & analysis
|-- requirements.txt       -> Dependencies
|-- README.txt             -> Project documentation
|
|-- models/                -> Saved ML models (ignored in Git)
|   |-- fraud_xgb_model.pkl
|   |-- feature_selector.pkl
|   |-- scaler.pkl
|
|-- data/                  -> Dataset (ignored in Git)
    |-- creditcard.csv


TECH STACK
----------
Programming Language : Python
Machine Learning     : XGBoost, Scikit-learn
Feature Selection    : RFECV
Visualization        : Streamlit, Plotly
Model Persistence    : Joblib
Version Control      : Git, GitHub


DASHBOARD FUNCTIONALITY
----------------------
- Displays key performance metrics:
  Precision, Recall, F1-score, ROC-AUC

- Interactive confusion matrix:
  Visual analysis of false positives and false negatives

- Feature importance chart:
  Shows most influential features for fraud detection

- Manual transaction fraud prediction:
  User inputs only top features including Amount
  Real-time fraud probability output


MODEL WORKFLOW
--------------
1. Data preprocessing and exploration
2. Feature selection using RFECV
3. Selective scaling of transaction amount
4. XGBoost model training
5. Model evaluation using ROC-AUC and confusion matrix
6. Deployment using Streamlit


BUSINESS VALUE
--------------
- Reduces financial losses due to fraud
- Supports data-driven risk decisions
- Enables non-technical stakeholders to interact with ML predictions
- Demonstrates production-ready ML deployment


MANUAL PREDICTION LOGIC
----------------------
- Users provide input for only the top 5 important features
- Transaction Amount is mandatory
- All other features are auto-filled using historical medians
- Same preprocessing pipeline is applied as during training


LEARNINGS AND CHALLENGES
-----------------------
- Handling highly imbalanced datasets
- Maintaining pipeline consistency during deployment
- Managing feature-scaler mismatches
- Resolving dependency compatibility issues


FUTURE IMPROVEMENTS
-------------------
- Fraud loss estimation (₹ saved)
- Model comparison (Logistic Regression vs XGBoost)
- SHAP-based explanations (offline)
- Streamlit Cloud deployment
- Enhanced UI for business users


AUTHOR
------
Name  : Naga Niroop
Degree: B.Tech - Artificial Intelligence
Role  : Final Year Student


LICENSE
-------
This project is intended for educational and demonstration purposes only.
