# LoanIQ — AI-Powered Loan Approval Prediction System

> An end-to-end ML web application that predicts loan approval with explainable AI (SHAP + LIME), role-based access (user/admin), full admin panel, activity log, user profiles, email verification, prediction history, EMI calculator, PDF reports, and an analytics dashboard.

---

## Video Demo

https://github.com/user-attachments/assets/e33417bf-384f-41e9-a908-f6ab358bc934

---

## Features

**Machine Learning**
- Random Forest / XGBoost / LightGBM — auto-selects best model via 5-fold cross-validation
- Feature engineering — EMI ratio, Total Income, Loan-Income Ratio, Income Per Dependent
- SHAP explainability — top 5 factors as itemized ledger entries
- LIME explanation — local feature effects alongside SHAP for double verification
- Risk score (0–100) and suggested maximum loan amount

**Authentication & Roles**
- User signup with email verification (one-time link sent via email)
- Login, logout, forgot password (email reset link), change password
- First registered account is automatically promoted to **Admin**
- Role-based navigation — admins and regular users see completely different interfaces
- Blocked users are stopped at login with a clear message

**Admin Panel** (`/admin`)
- System-wide stats — total users, total applications, approved/rejected counts, avg confidence
- Full searchable, filterable (all/active/blocked), paginated user registry
- Block / Unblock any non-admin user
- Delete a user and their entire prediction history (admins are protected)
- Export full user list as CSV
- Model retrain button — triggers RF/XGBoost/LightGBM cross-validation and hot-reloads best model
- Bell icon with unread notification badge and dropdown
- Link to Activity Log

**Activity Log** (`/admin/activity-log`)
- Records every login, logout, prediction, block, delete, and model retrain with timestamp and IP

**User Profile** (`/profile`)
- Account info — username, email, join date, email verification status
- Personal application stats — total filings, approved, rejected, avg confidence

**Applicant Features**
- Multi-step loan application form with progress bar
- Per-user prediction history with pagination (SQLite)
- Loan scenario comparison — compare 3 applications side by side
- EMI calculator with principal vs interest breakdown chart
- PDF report download — formatted decision report with SHAP factors
- CSV export of personal prediction history
- Email notification on every decision

**Analytics Dashboard** (`/dashboard`)
- Regular users see their own stats; admins see system-wide aggregated stats
- Approval/rejection doughnut (click to filter history)
- 30-day monthly trend line chart
- Feature importance bar chart

**Design**
- Case file / underwriting ledger visual identity throughout
- Decision stamp animation on result page (APPROVED / DECLINED rubber-stamp effect)
- SHAP and LIME displayed as accounting ledger entries (credit/debit)
- `Fraunces` (display) + `IBM Plex Mono` (data/ledger) typography
- Dark / Light theme toggle (persists via localStorage)
- Loading animation, page transitions, micro-interactions
- Fully responsive

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
│   ├── landing.html
│   ├── login.html
│   ├── signup.html
│   ├── forgot_password.html
│   ├── reset_password.html
│   ├── change_password.html
│   ├── profile.html
│   ├── index.html            ← applicant: loan form
│   ├── result.html           ← decision stamp + SHAP + LIME ledger
│   ├── history.html          ← own (user) / all (admin) with pagination
│   ├── dashboard.html        ← own (user) / system-wide (admin)
│   ├── calculator.html       ← EMI calculator
│   ├── compare.html          ← scenario comparison
│   ├── admin.html            ← system ledger + user registry
│   └── activity_log.html     ← admin audit trail
│
├── static/
│   ├── css/
│   │   ├── style.css
│   │   └── landing.css
│   └── js/
│       ├── form.js
│       └── theme.js
│
├── app.py
├── auth.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## Pages & Access

| URL | Access | Description |
|-----|--------|-------------|
| `/welcome` | Public | Landing page |
| `/signup` | Public | Create account |
| `/login` | Public | Sign in |
| `/verify-email/<token>` | Public | Email verification link |
| `/forgot-password` | Public | Request reset email |
| `/reset-password/<token>` | Public | Set new password |
| `/change-password` | User & Admin | Change own password |
| `/profile` | User only | Account info + application stats |
| `/` | User only | Loan application form |
| `/compare` | User only | Compare 3 scenarios |
| `/calculator` | User only | EMI calculator |
| `/history` | User & Admin | Own / all predictions with pagination |
| `/dashboard` | User & Admin | Own / system-wide analytics |
| `/export-csv` | User & Admin | CSV export scoped to role |
| `/download-report` | User only | PDF report of last prediction |
| `/admin` | Admin only | System ledger + user management |
| `/admin/activity-log` | Admin only | Audit trail |
| `/admin/notifications` | Admin only | Bell dropdown (JSON) |
| `/admin/retrain` | Admin only | Retrain model |

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

To enable emails (verification, reset, notifications), set Gmail credentials in `app.py`:
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
![LIME](https://img.shields.io/badge/LIME-Local%20Explanation-blueviolet)
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