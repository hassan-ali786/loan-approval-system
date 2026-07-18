# LoanIQ — AI-Powered Loan Approval System

A production-grade end-to-end machine learning web application that predicts loan eligibility with dual explainability (SHAP + LIME), role-based access control, a full admin panel, prediction history, EMI calculator, PDF reports, and an analytics dashboard.

---

## Demo

> Video demo coming soon after live deployment.

---

## Features

### Machine Learning
- **Auto model selection** — Random Forest, XGBoost, and LightGBM compete via 5-fold stratified cross-validation; the best-performing model is saved automatically
- **Feature engineering** — 4 derived features: Total Income, EMI ratio, Income Per Dependent, Loan-to-Income Ratio
- **SHAP explainability** — top 5 decision factors displayed as itemized ledger entries
- **LIME explanation** — local feature effects alongside SHAP for independent verification
- **Risk score** — model approval probability mapped to a 0–100 risk scale
- **Max loan suggestion** — calculates the applicant's ceiling based on the 40% EMI rule

### Authentication
- Signup with email verification (one-time token link)
- Login, logout, forgot password (email reset link), change password
- Passwords hashed with Werkzeug (bcrypt-backed)
- First registered account is automatically promoted to admin
- Blocked users are rejected at login with a clear message

### Admin Panel
- System-wide stats — total users, applications, approval/rejection counts, average confidence
- Searchable, filterable (all / active / blocked), paginated user registry
- Block / unblock or delete any non-admin user
- Export full user list as CSV
- Model retrain button — triggers cross-validation pipeline and hot-reloads the best model in memory
- Bell icon with unread notification badge and dropdown
- Activity log — every login, logout, prediction, block, delete, and retrain recorded with IP and timestamp

### Applicant Features
- 3-step loan application form with animated step indicator
- Per-user prediction history with pagination
- Loan scenario comparison — run 3 scenarios side by side
- EMI calculator with principal vs interest doughnut chart
- PDF report download (ReportLab) — decision, SHAP factors, application details
- CSV export of personal prediction history
- Email notification on every decision (Flask-Mail)

### Analytics Dashboard
- Regular users see their own stats; admins see system-wide aggregated data
- Approval/rejection doughnut chart — click a segment to filter history
- 30-day trend line chart
- Feature importance horizontal bar chart

### User Profile
- Account info, join date, email verification status
- Personal application stats

### UI
- Sidebar navigation layout — fixed left sidebar with section groups and active states
- **Inter** (UI) + **JetBrains Mono** (data/code) typography
- Dark / Light theme toggle persisting via localStorage
- Decision stamp animation on result page
- Loading overlay on form submission
- Fully responsive — mobile sidebar slides in via hamburger button

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~82% |
| Recall (Approved) | ~0.89 |
| Cross-validation | 5-fold Stratified |
| Model | Auto-selected (RF / XGBoost / LightGBM) |
| Total Features | 15 (11 original + 4 engineered) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | Flask 3.x |
| Authentication | Flask-Login, Werkzeug |
| Email | Flask-Mail (Gmail SMTP) |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP, LIME |
| Data | pandas, NumPy |
| Database | SQLite (via Python sqlite3) |
| PDF Generation | ReportLab |
| Charts | Chart.js |
| Production Server | Gunicorn |

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
│   ├── loan_model.pkl          # trained model (auto-selected)
│   └── features.pkl            # feature list
│
├── templates/
│   ├── base_user.html          # sidebar layout base — all applicant pages extend this
│   ├── base_admin.html         # sidebar layout base — all admin pages extend this
│   ├── landing.html            # public landing page
│   ├── login.html
│   ├── signup.html
│   ├── forgot_password.html
│   ├── reset_password.html
│   ├── change_password.html
│   ├── profile.html
│   ├── index.html              # loan application form
│   ├── result.html             # decision + SHAP + LIME
│   ├── history.html            # own (user) / all (admin) with pagination
│   ├── dashboard.html          # own (user) / system-wide (admin)
│   ├── calculator.html         # EMI calculator
│   ├── compare.html            # scenario comparison
│   ├── admin.html              # user registry + system stats
│   └── activity_log.html       # audit trail
│
├── static/
│   ├── css/
│   │   ├── style.css           # full design system
│   │   └── landing.css         # landing page styles
│   └── js/
│       ├── form.js             # multi-step form + step dots
│       └── theme.js            # dark/light toggle + mobile sidebar
│
├── app.py                      # Flask routes, prediction pipeline, admin, PDF
├── auth.py                     # User model, password hashing, tokens, activity log
├── train_model.py              # training pipeline with model comparison
├── requirements.txt
└── README.md
```

---

## Pages

| URL | Access | Description |
|-----|--------|-------------|
| `/welcome` | Public | Landing page |
| `/signup` | Public | Create account |
| `/login` | Public | Sign in |
| `/verify-email/<token>` | Public | Email verification |
| `/forgot-password` | Public | Request reset email |
| `/reset-password/<token>` | Public | Set new password |
| `/` | User | Loan application form |
| `/compare` | User | Compare 3 scenarios |
| `/calculator` | User | EMI calculator |
| `/profile` | User | Account info and stats |
| `/change-password` | User + Admin | Update password |
| `/history` | User + Admin | Predictions (scoped by role) |
| `/dashboard` | User + Admin | Analytics (scoped by role) |
| `/export-csv` | User + Admin | CSV export (scoped by role) |
| `/download-report` | User | PDF report of last prediction |
| `/admin` | Admin | System ledger + user management |
| `/admin/activity-log` | Admin | Audit trail |
| `/admin/retrain` | Admin | Retrain model |
| `/admin/notifications` | Admin | Bell dropdown (JSON) |

---

## Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/hassan-ali786/loan-approval-prediction.git
cd loan-approval-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train_model.py
```
This runs RF vs XGBoost vs LightGBM comparison, prints accuracy/recall, and saves the best model to `model/`.

**4. Run the app**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000
```

The first account you register becomes the admin automatically.

---

## Email Configuration (Optional)

To enable email verification, password reset, and loan decision notifications, add your Gmail credentials to `app.py`:

```python
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your-app-password'   # Gmail App Password
```

The app works without this — emails fail silently and everything else functions normally.

---

## Roadmap

- [ ] Docker containerization
- [ ] Live deployment (Render)
- [ ] PostgreSQL migration for production scale
- [ ] API endpoints for external integration

---

## Dataset

Kaggle — Loan Prediction Problem Dataset
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

---

## Author

**Hassan Ali** — Data Scientist & ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/hassan-ali-datascientist)
[![GitHub](https://img.shields.io/badge/GitHub-hassan--ali786-181717?logo=github&logoColor=white)](https://github.com/hassan-ali786)
