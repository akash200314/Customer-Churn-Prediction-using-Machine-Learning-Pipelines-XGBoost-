🧠 Customer Churn Prediction using Machine Learning Pipelines (XGBoost)
📋 Project Overview

This project focuses on predicting customer churn using machine learning pipelines.
By analyzing customer demographics, subscription details, and usage behavior, we aim to identify customers who are likely to discontinue services.
The solution uses Scikit-learn pipelines and XGBoost for robust preprocessing, model training, and deployment.

🎯 Objective

To build a machine learning model that accurately predicts customer churn, enabling proactive retention strategies.

🧰 Tech Stack

Programming Language: Python

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, joblib

Environment: Jupyter Notebook

Model Used: XGBoost Classifier

📊 Dataset Description

The project uses two CSV files:

customer_churn_training.csv → for training the model

customer_churn_testing.csv → for evaluating model performance

Key Columns:

Feature	Description
Gender	Customer gender (Male/Female)
Subscription Type	Plan type (Basic, Premium, etc.)
Contract Length	Duration of subscription
Monthly Charges	Customer’s monthly payment
Total Spend	Total amount spent
Churn	Target variable (1 = churned, 0 = retained)
⚙️ Workflow

Data Loading – Import training and testing datasets.

Data Cleaning – Handle missing values, remove irrelevant columns, and encode target variable.

Feature Engineering – Scale numerical features and one-hot encode categorical ones.

Model Building – Use XGBoost within a Scikit-learn pipeline.

Model Evaluation – Evaluate using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

Feature Importance – Visualize the top features influencing churn.

Model Saving – Export trained model using joblib for future inference.

Predictions Export – Save predictions in an Excel file for analysis.

🧩 Key Features

Fully automated data preprocessing and transformation pipeline

High-performance model using XGBoost with cross-validation

Confusion matrix and feature importance visualization

Model and results saved for easy deployment or analysis

📈 Model Performance 
Metric	Score

Accuracy	0.504

Precision	0.489

Recall	0.998

F1 Score	0.656

ROC-AUC	0.727

<img width="468" height="396" alt="download" src="https://github.com/user-attachments/assets/72813d0f-e82f-4b18-94b8-7b64ec8ba64f" />

<img width="844" height="529" alt="download" src="https://github.com/user-attachments/assets/3431f766-b598-4e37-b00d-cc778592cd16" />



📂 Outputs

customer_churn_xgb_pipeline.joblib → Saved trained model

customer_churn_predictions.xlsx → Excel file with predictions and probabilities

🚀 How to Run

Open the notebook: Customer_Churn_Prediction_Pipeline.ipynb

Place both CSVs in the same directory:

customer_churn_training.csv

customer_churn_testing.csv

Run all notebook cells sequentially.

View results, feature importance, and saved files in the project folder.

🧠 Insights

Subscription type and contract length significantly impact churn probability.

High monthly charges correlate with higher churn likelihood.

The model helps businesses design targeted retention campaigns.

🏁 Conclusion
The project successfully demonstrates a complete end-to-end churn prediction pipeline — from raw data to actionable insights — using modern machine learning techniques.
