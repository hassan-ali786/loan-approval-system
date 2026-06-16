from flask import Flask, request, render_template, jsonify, make_response, session
import pickle
import numpy as np
import pandas as pd
import sqlite3
import shap
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

app = Flask(__name__)
app.secret_key = "loaniq-secret-key"

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

def get_shap_factors(X_input):
    """Safely extract SHAP values regardless of model type."""
    shap_values = explainer.shap_values(X_input)
    if isinstance(shap_values, list):
        # Random Forest: list of [class0_array, class1_array]
        sv_raw = np.array(shap_values[1]).flatten()
    else:
        # XGBoost / LightGBM: single array
        sv_raw = np.array(shap_values).flatten()
    sv_floats = [float(v) for v in sv_raw[:len(FEATURES)]]
    pairs = sorted(zip(FEATURES, sv_floats), key=lambda x: abs(x[1]), reverse=True)[:5]
    return [{"feature": k, "value": round(v, 4), "width": min(round(abs(v) * 300, 1), 100)} for k, v in pairs]

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

            total_income      = applicant_income + coapplicant_income
            emi               = loan_amount / loan_term if loan_term > 0 else 0
            income_per_dep    = total_income / (dependents + 1)
            loan_income_ratio = loan_amount / (total_income + 1)

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

            X_input = pd.DataFrame([[features_dict[f] for f in FEATURES]], columns=FEATURES)

            prediction  = model.predict(X_input)[0]
            proba_array = model.predict_proba(X_input)[0]
            probability = round(float(proba_array[1]) * 100, 1) if prediction == 1 else round(float(proba_array[0]) * 100, 1)
            result      = "Approved" if prediction == 1 else "Rejected"

            top_factors = get_shap_factors(X_input)

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

            # Store in session for PDF download
            session['last_result'] = {
                'result':      result,
                'probability': probability,
                'top_factors': top_factors,
                'income':      applicant_income,
                'co_income':   coapplicant_income,
                'loan_amount': loan_amount,
                'loan_term':   loan_term,
                'credit':      features_dict['Credit_History'],
                'timestamp':   datetime.now().strftime("%Y-%m-%d %H:%M")
            }

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

    importances = model.feature_importances_
    fi_pairs  = sorted(zip(FEATURES, importances.tolist()), key=lambda x: x[1], reverse=True)[:10]
    fi_labels = [f[0] for f in fi_pairs]
    fi_values = [round(f[1] * 100, 2) for f in fi_pairs]

    return render_template("dashboard.html",
        approved=approved,
        rejected=rejected,
        avg_prob=avg_prob,
        avg_loan=avg_loan,
        fi_labels=json.dumps(fi_labels),
        fi_values=json.dumps(fi_values)
    )


@app.route("/calculator")
def calculator():
    return render_template("calculator.html")


@app.route("/download-report")
def download_report():
    data = session.get('last_result')
    if not data:
        return "No result found. Please submit an application first.", 400

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)

    styles  = getSampleStyleSheet()
    title_s = ParagraphStyle('title', fontSize=22, fontName='Helvetica-Bold',
                              textColor=colors.HexColor('#0D1B2A'), spaceAfter=6)
    sub_s   = ParagraphStyle('sub', fontSize=11, fontName='Helvetica',
                              textColor=colors.HexColor('#5A6A7A'), spaceAfter=20)
    head_s  = ParagraphStyle('head', fontSize=13, fontName='Helvetica-Bold',
                              textColor=colors.HexColor('#0D1B2A'), spaceBefore=14, spaceAfter=6)

    result_color = colors.HexColor('#27ae60') if data['result'] == 'Approved' else colors.HexColor('#e74c3c')

    elements = [
        Paragraph("LoanIQ — Loan Decision Report", title_s),
        Paragraph(f"Generated: {data['timestamp']}", sub_s),
        Spacer(1, 0.3*cm),
        Paragraph("Decision Summary", head_s),
        Table([
            ["Decision",        data['result']],
            ["Confidence Score", f"{data['probability']}%"],
        ], colWidths=[6*cm, 10*cm],
        style=TableStyle([
            ('BACKGROUND',     (0,0), (0,-1), colors.HexColor('#F0F4F8')),
            ('FONTNAME',       (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE',       (0,0), (-1,-1), 11),
            ('TEXTCOLOR',      (1,0), (1,0), result_color),
            ('FONTNAME',       (1,0), (1,0), 'Helvetica-Bold'),
            ('FONTSIZE',       (1,0), (1,0), 13),
            ('GRID',           (0,0), (-1,-1), 0.5, colors.HexColor('#D0D8E4')),
            ('PADDING',        (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#F9FBFF')]),
        ])),
        Spacer(1, 0.4*cm),
        Paragraph("Application Details", head_s),
        Table([
            ["Applicant Income",    str(data['income'])],
            ["Co-applicant Income", str(data['co_income'])],
            ["Loan Amount",         f"{data['loan_amount']}k"],
            ["Loan Term",           f"{data['loan_term']} months"],
            ["Credit History",      "Good" if data['credit'] == 1 else "Poor"],
        ], colWidths=[6*cm, 10*cm],
        style=TableStyle([
            ('BACKGROUND',     (0,0), (0,-1), colors.HexColor('#F0F4F8')),
            ('FONTNAME',       (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE',       (0,0), (-1,-1), 11),
            ('GRID',           (0,0), (-1,-1), 0.5, colors.HexColor('#D0D8E4')),
            ('PADDING',        (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#F9FBFF')]),
        ])),
        Spacer(1, 0.4*cm),
        Paragraph("Top Decision Factors (SHAP)", head_s),
        Table(
            [["Feature", "SHAP Value", "Direction"]] +
            [[f['feature'], str(f['value']), "Positive" if f['value'] > 0 else "Negative"]
             for f in data['top_factors']],
            colWidths=[7*cm, 4*cm, 5*cm],
            style=TableStyle([
                ('BACKGROUND',     (0,0), (-1,0), colors.HexColor('#0D1B2A')),
                ('TEXTCOLOR',      (0,0), (-1,0), colors.white),
                ('FONTNAME',       (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',       (0,0), (-1,-1), 10),
                ('GRID',           (0,0), (-1,-1), 0.5, colors.HexColor('#D0D8E4')),
                ('PADDING',        (0,0), (-1,-1), 8),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9FBFF')]),
            ])
        ),
        Spacer(1, 0.6*cm),
        Paragraph("This report is generated by LoanIQ — an AI-powered loan prediction system.",
                  ParagraphStyle('footer', fontSize=9, textColor=colors.HexColor('#8A9BB0')))
    ]

    doc.build(elements)
    buffer.seek(0)

    response = make_response(buffer.read())
    response.headers['Content-Type']        = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=LoanIQ_Report.pdf'
    return response


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)