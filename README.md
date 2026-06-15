# LoanIQ — Loan Approval Prediction System

> An end-to-end ML web application that predicts loan approval with explainable AI, prediction history, and an analytics dashboard.

---

## Video Demo

https://github.com/user-attachments/assets/e33417bf-384f-41e9-a908-f6ab358bc934

---

## Features

- **XGBoost Model** — upgraded from Random Forest with feature engineering (EMI, Total Income, Loan-Income Ratio)
- **SHAP Explainability** — shows top 5 factors behind every decision
- **Confidence Score** — model probability displayed on result page
- **Prediction History** — every application saved to SQLite, viewable at `/history`
- **Analytics Dashboard** — approved vs rejected counts, avg confidence, avg loan amount with doughnut chart at `/dashboard`
- **Multi-step Form** — 3-step form with progress bar and error handling
- **Premium Fintech UI** — dark navy + gold theme, fully responsive

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~82% |
| Recall (Approved) | ~0.89 |
| Model | XGBoost |
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
│   ├── loan_model.pkl
│   └── features.pkl
│
├── templates/
│   ├── index.html       ← multi-step application form
│   ├── result.html      ← prediction result + SHAP factors
│   ├── history.html     ← past predictions table
│   └── dashboard.html   ← analytics dashboard
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── form.js
│
├── app.py               ← Flask routes + SHAP + SQLite
├── train_model.py       ← XGBoost training pipeline
├── history.db           ← auto-created SQLite database
├── requirements.txt
└── README.md
```

---

## Pages

| URL | Description |
|-----|-------------|
| `/` | Loan application form |
| `/history` | Last 50 predictions |
| `/dashboard` | Analytics + charts |

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
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey?logo=sqlite)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas)
![Chart.js](https://img.shields.io/badge/Chart.js-Dashboard-red)
![HTML](https://img.shields.io/badge/HTML-Frontend-red?logo=html5)
![CSS](https://img.shields.io/badge/CSS-Styling-blue?logo=css3)

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