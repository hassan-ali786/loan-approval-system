Loan Approval Prediction System
Overview

The Loan Approval Prediction System is an end-to-end machine learning web application that predicts whether a loan application will be approved or rejected based on user input.

This project demonstrates a complete workflow including data preprocessing, model training, model deployment, and real-time prediction using a web interface.

It simulates a basic decision-support system used in banking and fintech environments.

Features
Machine Learning Model
Uses a trained Random Forest Classifier for prediction
Interactive Web Application
Simple and user-friendly form for input
Real-Time Prediction
Instant approval or rejection result
Data Preprocessing
Handles missing values and categorical encoding
Professional UI
Clean and responsive design
End-to-End Pipeline
Dataset → Model Training → Model Saving → Web App → Prediction
Dataset

This project uses a public dataset from Kaggle:

https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

Features
Gender
Marital Status
Dependents
Education
Employment Status
Applicant Income
Co-Applicant Income
Loan Amount
Loan Term
Credit History
Property Area

Project Structure
Loan_Approval_System/
│
├── app.py
├── train_model.py
│
├── dataset/
│   └── loan_data.csv
│
├── model/
│   └── loan_model.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── form.js
│
├── requirements.txt
└── README.md
