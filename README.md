# LoanIQ — AI-Powered Loan Approval Prediction System

> An end-to-end ML web application that predicts loan approval with explainable AI, prediction history, EMI calculator, PDF reports, and an analytics dashboard.

---

## Video Demo

https://github.com/user-attachments/assets/e33417bf-384f-41e9-a908-f6ab358bc934

---

## Features

- **XGBoost / LightGBM / Random Forest** — auto-selects best model via 5-fold cross-validation
- **Feature Engineering** — EMI ratio, Total Income, Loan-Income Ratio, Income Per Dependent
- **SHAP Explainability** — top 5 factors behind every decision with visual bars
- **Confidence Score** — model probability on every result
- **PDF Report Download** — formatted loan decision report with SHAP factors
- **EMI Calculator** — monthly installment calculator with principal vs interest chart
- **Prediction History** — every application saved to SQLite, viewable at `/history`
- **Analytics Dashboard** — approved/rejected stats, avg confidence, feature importance chart at `/dashboard`
- **Dark / Light Theme Toggle** — persists across sessions via localStorage
- **Loading Animation** — spinner on form submit
- **Multi-step Form** — 3-step form with progress bar and error handling
- **Premium Fintech UI** — dark navy + gold theme, fully responsive

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~82% |
| Recall (Approved) | ~0.89 |
| CV Folds | 5-fold Stratified |
| Best Model | Auto-selected (RF / XGBoost / LightGBM) |
| Features | 15 (11 original + 4 engineered) |

---

## Engineered Features

| Feature | Formula |
|---------|---------|
| `TotalIncome` | ApplicantIncome + CoapplicantIncome |
| `EMI` | LoanAmount / Loan_Amount_Term |
| `IncomePerDependent` | TotalIncome / (Dependents + 1) |
| `LoanIncomeRatio` | LoanAmount / (TotalIncome + 1) |

---

## Project Structure

```
Loan-Approval-System/
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── model/
│   ├── loan_model.pkl        ← best model (auto-selected)
│   └── features.pkl          ← feature list
│
├── templates/
│   ├── index.html            ← multi-step application form
│   ├── result.html           ← prediction result + SHAP + PDF download
│   ├── history.html          ← past predictions table
│   ├── dashboard.html        ← analytics + feature importance chart
│   └── calculator.html       ← EMI calculator
│
├── static/
│   ├── css/
│   │   └── style.css         ← dark/light theme
│   └── js/
│       ├── form.js           ← multi-step form logic
│       └── theme.js          ← theme toggle + loading animation
│
├── app.py                    ← Flask routes, SHAP, SQLite, PDF
├── train_model.py            ← RF vs XGBoost vs LightGBM + CV
├── history.db                ← auto-created SQLite database
├── requirements.txt
└── README.md
```

---

## Pages

| URL | Description |
|-----|-------------|
| `/` | Loan application form |
| `/history` | Last 50 predictions |
| `/dashboard` | Analytics + feature importance chart |
| `/calculator` | EMI calculator |
| `/download-report` | PDF report of last prediction |

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/hassan-ali786/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Create Virtual Environment (Optional)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python train_model.py
```

### 5. Run the App
```bash
python app.py
```

### 6. Open in Browser
```
http://127.0.0.1:5000
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Model-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey?logo=sqlite)
![ReportLab](https://img.shields.io/badge/ReportLab-PDF%20Reports-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas)
![Chart.js](https://img.shields.io/badge/Chart.js-Dashboard-red)

---

## Dataset

Kaggle — Loan Prediction Problem Dataset  
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

---

## Author

**Hassan Ali** — Data Scientist & ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/hassan-ali-datascientist)
[![GitHub](https://img.shields.io/badge/GitHub-hassan--ali786-black?logo=github)](https://github.com/hassan-ali786)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-gold)](https://hassanali-portfolio.vercel.app)