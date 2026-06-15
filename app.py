from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import sqlite3
import shap
import json
from datetime import datetime

app = Flask(__name__)

# Load model and features
model    = pickle.load(open("model/loan_model.pkl", "rb"))
FEATURES = pickle.load(open("model/features.pkl", "rb"))

# SHAP explainer
explainer = shap.TreeExplainer(model)

# =========================
# SQLite Setup
# =========================
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            result TEXT,
            probability REAL,
            applicant_income REAL,
            loan_amount REAL,
            credit_history INTEGER,
            top_factors TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            applicant_income   = float(request.form.get('ApplicantIncome', 0))
            coapplicant_income = float(request.form.get('CoapplicantIncome', 0))
            loan_amount        = float(request.form.get('LoanAmount', 150))
            loan_term          = float(request.form.get('Loan_Amount_Term', 360))
            dependents         = int(request.form.get('Dependents', 0))

            total_income        = applicant_income + coapplicant_income
            emi                 = loan_amount / loan_term if loan_term > 0 else 0
            income_per_dep      = total_income / (dependents + 1)
            loan_income_ratio   = loan_amount / (total_income + 1)

            features_dict = {
                'Gender':             int(request.form.get('Gender', 1)),
                'Married':            int(request.form.get('Married', 1)),
                'Dependents':         dependents,
                'Education':          int(request.form.get('Education', 1)),
                'Self_Employed':      int(request.form.get('Self_Employed', 0)),
                'ApplicantIncome':    applicant_income,
                'CoapplicantIncome':  coapplicant_income,
                'LoanAmount':         loan_amount,
                'Loan_Amount_Term':   loan_term,
                'Credit_History':     int(request.form.get('Credit_History', 1)),
                'Property_Area':      int(request.form.get('Property_Area', 2)),
                'TotalIncome':        total_income,
                'EMI':                emi,
                'IncomePerDependent': income_per_dep,
                'LoanIncomeRatio':    loan_income_ratio
            }

            X_input = np.array([[features_dict[f] for f in FEATURES]])

            prediction   = model.predict(X_input)[0]
            proba_array  = model.predict_proba(X_input)[0]
            probability  = round(float(proba_array[1]) * 100, 1) if prediction == 1 else round(float(proba_array[0]) * 100, 1)
            result       = "Approved" if prediction == 1 else "Rejected"

            # SHAP values
            shap_values  = explainer.shap_values(X_input)
            sv           = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
            shap_pairs   = sorted(zip(FEATURES, sv.tolist()), key=lambda x: abs(x[1]), reverse=True)[:5]
            top_factors  = [{"feature": k, "value": round(v, 4)} for k, v in shap_pairs]

            # Save to DB
            conn = sqlite3.connect("history.db")
            c    = conn.cursor()
            c.execute('''
                INSERT INTO predictions (timestamp, result, probability, applicant_income, loan_amount, credit_history, top_factors)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                result, probability,
                applicant_income, loan_amount,
                features_dict['Credit_History'],
                json.dumps(top_factors)
            ))
            conn.commit()
            conn.close()

            return render_template("result.html",
                result=result,
                probability=probability,
                top_factors=top_factors
            )

        except Exception as e:
            print("ERROR:", e)
            return render_template("index.html", error="Something went wrong. Please check your inputs.")

    return render_template("index.html")


@app.route("/history")
def history():
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT result, COUNT(*) FROM predictions GROUP BY result")
    counts = dict(c.fetchall())
    c.execute("SELECT AVG(probability) FROM predictions")
    avg_prob = round(c.fetchone()[0] or 0, 1)
    c.execute("SELECT AVG(loan_amount) FROM predictions")
    avg_loan = round(c.fetchone()[0] or 0, 1)
    conn.close()

    approved = counts.get("Approved", 0)
    rejected = counts.get("Rejected", 0)

    return render_template("dashboard.html",
        approved=approved,
        rejected=rejected,
        avg_prob=avg_prob,
        avg_loan=avg_loan
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)