# LoanIQ — AI-Powered Loan Approval Prediction System

> An end-to-end ML web application that predicts loan approval with explainable AI, role-based access (user/admin), prediction history, EMI calculator, PDF reports, and an analytics dashboard.

---

## Video Demo

https://github.com/user-attachments/assets/e33417bf-384f-41e9-a908-f6ab358bc934

---

## Features

**Machine Learning**
- Random Forest / XGBoost / LightGBM — auto-selects best model via 5-fold cross-validation
- Feature engineering — EMI ratio, Total Income, Loan-Income Ratio, Income Per Dependent
- SHAP explainability — top 5 factors behind every decision, shown as ledger entries
- Risk score (0–100) and suggested maximum loan amount

**Authentication & Roles**
- User authentication — signup, login, logout (Flask-Login, hashed passwords)
- First registered account is automatically promoted to **Admin**
- Role-based navigation — admins and regular users see different menus and pages
- Admins are redirected away from Apply / Compare / EMI Calculator (those are applicant-only tools)
- Change password (any logged-in user) and forgot-password email reset flow

**Admin Panel** (`/admin`)
- System-wide stats — total users, total applications, approved/rejected counts, avg confidence
- Full user registry — role, status, application count, join date
- Block / Unblock any non-admin user
- Delete a user and their entire prediction history (admins are protected from deletion)
- Export full user list as CSV
- Blocked users are stopped at login with a clear message

**Applicant Features**
- Multi-step loan application form with progress bar
- Per-user prediction history (SQLite)
- Loan scenario comparison — compare 3 applications side by side
- EMI calculator with principal vs interest breakdown
- PDF report download — formatted decision report with SHAP factors
- CSV export of personal prediction history
- Email notification on every decision (Flask-Mail)

**Analytics Dashboard** (`/dashboard`)
- Regular users see their own approval breakdown, trend, and stats
- Admins see the same dashboard aggregated across **all** users
- Approval/rejection doughnut chart (click a slice to filter history)
- 30-day monthly trend line chart
- Feature importance bar chart

**Design**
- A distinctive "case file" visual identity throughout — decisions are presented
  as an underwriting stamp, SHAP factors as ledger entries, and the admin panel
  as a "system ledger" with role tags and status indicators
- `Fraunces` (display) + `IBM Plex Mono` (data/ledger) typography
- Dark / Light theme toggle (persists via localStorage)
- Loading animation on submission, page-load transitions, micro-interactions
- Fully responsive (mobile, tablet, desktop)

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
│   ├── landing.html          ← public welcome page
│   ├── login.html
│   ├── signup.html
│   ├── forgot_password.html
│   ├── reset_password.html
│   ├── change_password.html
│   ├── index.html            ← applicant-only: loan application form
│   ├── result.html           ← decision stamp + SHAP ledger
│   ├── history.html          ← own history (user) / all history (admin)
│   ├── dashboard.html        ← own stats (user) / system-wide stats (admin)
│   ├── calculator.html       ← applicant-only: EMI calculator
│   ├── compare.html          ← applicant-only: scenario comparison
│   └── admin.html            ← admin-only: system ledger / user management
│
├── static/
│   ├── css/
│   │   ├── style.css         ← core design system + case-file/ledger theme
│   │   └── landing.css
│   └── js/
│       ├── form.js
│       └── theme.js
│
├── app.py                    ← Flask routes, SHAP, SQLite, auth, admin, email, PDF
├── auth.py                   ← User model, password hashing, reset tokens, admin helpers
├── train_model.py            ← RF vs XGBoost vs LightGBM + cross-validation
├── requirements.txt
└── README.md
```

---

## Pages & Access

| URL | Access | Description |
|-----|--------|--------------|
| `/welcome` | Public | Landing page |
| `/signup` | Public | Create account |
| `/login` | Public | Sign in |
| `/forgot-password` | Public | Request password reset email |
| `/reset-password/<token>` | Public (one-time) | Set new password from email link |
| `/change-password` | User & Admin | Change your own password |
| `/` | User only | Loan application form |
| `/compare` | User only | Compare 3 loan scenarios |
| `/calculator` | User only | EMI calculator |
| `/history` | User & Admin | Own history (user) / all applications (admin) |
| `/dashboard` | User & Admin | Own stats (user) / system-wide stats (admin) |
| `/admin` | Admin only | System ledger — block, delete, export users |
| `/download-report` | User only | PDF report of last prediction |
| `/export-csv` | User & Admin | CSV export, scoped to role |

---

## Local Setup

```bash
git clone https://github.com/hassan-ali786/loan-approval-prediction.git
cd loan-approval-prediction
pip install -r requirements.txt
python train_model.py
python app.py
```
Open `http://127.0.0.1:5000`

The **first account you sign up** automatically becomes the admin.

To enable email notifications and password-reset emails, set your Gmail credentials in `app.py`:
```python
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your-app-password'
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)
![Flask-Login](https://img.shields.io/badge/Flask--Login-Auth-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Model-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey?logo=sqlite)
![ReportLab](https://img.shields.io/badge/ReportLab-PDF%20Reports-red)
![Chart.js](https://img.shields.io/badge/Chart.js-Dashboard-red)

---

## Roadmap

- [ ] Containerize with Docker
- [ ] Deploy live demo (Render)

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