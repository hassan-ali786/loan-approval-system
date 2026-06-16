from flask import Flask, request, render_template, jsonify, make_response, session
import pickle
import numpy as np
import pandas as pd
import sqlite3
import shap
import json
import csv
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = Flask(__name__)
app.secret_key = "loaniq-secret-key"

# ─────────────────────────────────────────
# Load model & feature list
# ─────────────────────────────────────────
model    = pickle.load(open("model/loan_model.pkl", "rb"))
FEATURES = pickle.load(open("model/features.pkl", "rb"))

# SHAP explainer — built once at startup for speed
explainer = shap.TreeExplainer(model)


# ─────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────
def init_db():
    """Create predictions table if it doesn't exist."""
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            result           TEXT,
            probability      REAL,
            risk_score       INTEGER,
            applicant_income REAL,
            loan_amount      REAL,
            credit_history   INTEGER,
            top_factors      TEXT,
            max_loan         REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()


# ─────────────────────────────────────────
# ML helpers
# ─────────────────────────────────────────
def get_shap_factors(X_input):
    """
    Extract top-5 SHAP factors safely for any tree model.
    Random Forest returns a list [class0, class1]; others return a single array.
    """
    shap_values = explainer.shap_values(X_input)
    if isinstance(shap_values, list):
        sv_raw = np.array(shap_values[1]).flatten()   # class-1 (Approved)
    else:
        sv_raw = np.array(shap_values).flatten()

    sv_floats = [float(v) for v in sv_raw[:len(FEATURES)]]
    pairs = sorted(zip(FEATURES, sv_floats), key=lambda x: abs(x[1]), reverse=True)[:5]
    return [
        {
            "feature": k,
            "value":   round(v, 4),
            "width":   min(round(abs(v) * 300, 1), 100)
        }
        for k, v in pairs
    ]


def compute_risk_score(proba_approved):
    """
    Convert approval probability to a 0-100 risk score.
    Higher score = lower risk = more likely to be approved.
    """
    return round(proba_approved * 100)


def suggest_max_loan(total_income, loan_term, credit_history):
    """
    Suggest maximum loan amount based on income and credit.
    Rule: max EMI should not exceed 40% of monthly income.
    Adjusted downward for poor credit history.
    """
    monthly_income = total_income / 12
    max_emi        = monthly_income * 0.40
    r              = 0.085 / 12          # assumed 8.5% annual interest rate
    n              = loan_term

    if r > 0 and n > 0:
        max_loan = max_emi * (1 - (1 + r) ** (-n)) / r
    else:
        max_loan = max_emi * n

    if credit_history == 0:
        max_loan *= 0.6   # reduce by 40% for poor credit

    return round(max_loan / 1000, 1)    # return in thousands


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # ── Parse form inputs ──────────────────────
            applicant_income   = float(request.form.get('ApplicantIncome', 0))
            coapplicant_income = float(request.form.get('CoapplicantIncome', 0))
            loan_amount        = float(request.form.get('LoanAmount', 150))
            loan_term          = float(request.form.get('Loan_Amount_Term', 360))
            dependents         = int(request.form.get('Dependents', 0))
            credit_history     = int(request.form.get('Credit_History', 1))

            # ── Feature engineering ────────────────────
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
                'Credit_History':     credit_history,
                'Property_Area':      int(request.form.get('Property_Area', 2)),
                'TotalIncome':        total_income,
                'EMI':                emi,
                'IncomePerDependent': income_per_dep,
                'LoanIncomeRatio':    loan_income_ratio
            }

            # ── Model prediction ───────────────────────
            X_input     = pd.DataFrame([[features_dict[f] for f in FEATURES]], columns=FEATURES)
            prediction  = model.predict(X_input)[0]
            proba_array = model.predict_proba(X_input)[0]
            proba_approved = float(proba_array[1])
            probability    = round(proba_approved * 100, 1) if prediction == 1 else round(float(proba_array[0]) * 100, 1)
            result         = "Approved" if prediction == 1 else "Rejected"

            # ── Risk score & loan suggestion ───────────
            risk_score = compute_risk_score(proba_approved)
            max_loan   = suggest_max_loan(total_income, loan_term, credit_history)

            # ── SHAP top factors ───────────────────────
            top_factors = get_shap_factors(X_input)

            # ── Persist to SQLite ──────────────────────
            conn = sqlite3.connect("history.db")
            c    = conn.cursor()
            c.execute('''
                INSERT INTO predictions
                    (timestamp, result, probability, risk_score,
                     applicant_income, loan_amount, credit_history, top_factors, max_loan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                result, probability, risk_score,
                applicant_income, loan_amount, credit_history,
                json.dumps(top_factors), max_loan
            ))
            conn.commit()
            conn.close()

            # ── Store in session for PDF ───────────────
            session['last_result'] = {
                'result':      result,
                'probability': probability,
                'risk_score':  risk_score,
                'max_loan':    max_loan,
                'top_factors': top_factors,
                'income':      applicant_income,
                'co_income':   coapplicant_income,
                'loan_amount': loan_amount,
                'loan_term':   loan_term,
                'credit':      credit_history,
                'timestamp':   datetime.now().strftime("%Y-%m-%d %H:%M")
            }

            return render_template("result.html",
                result=result,
                probability=probability,
                risk_score=risk_score,
                max_loan=max_loan,
                top_factors=top_factors
            )

        except Exception as e:
            print("ERROR:", e)
            return render_template("index.html", error="Something went wrong. Please check your inputs.")

    return render_template("index.html")


@app.route("/history")
def history():
    """Show last 50 predictions from SQLite."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


@app.route("/export-csv")
def export_csv():
    """Export full prediction history as a downloadable CSV file."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT id, timestamp, result, probability, risk_score, applicant_income, loan_amount, credit_history, max_loan FROM predictions ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    # CSV header
    writer.writerow(["ID", "Timestamp", "Result", "Confidence (%)", "Risk Score",
                     "Applicant Income", "Loan Amount (k)", "Credit History", "Max Loan (k)"])
    for row in rows:
        writer.writerow(row)

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=LoanIQ_History.csv"
    response.headers["Content-Type"]        = "text/csv"
    return response


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard with approval stats, trend data, and feature importance."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()

    # Approval counts
    c.execute("SELECT result, COUNT(*) FROM predictions GROUP BY result")
    counts = dict(c.fetchall())

    # Averages
    c.execute("SELECT AVG(probability) FROM predictions")
    avg_prob = round(c.fetchone()[0] or 0, 1)

    c.execute("SELECT AVG(loan_amount) FROM predictions")
    avg_loan = round(c.fetchone()[0] or 0, 1)

    # Monthly trend — count per day for last 30 days
    c.execute("""
        SELECT DATE(timestamp) as day, result, COUNT(*) as cnt
        FROM predictions
        WHERE timestamp >= DATE('now', '-30 days')
        GROUP BY day, result
        ORDER BY day ASC
    """)
    trend_rows = c.fetchall()
    conn.close()

    approved = counts.get("Approved", 0)
    rejected = counts.get("Rejected", 0)

    # Build trend data for Chart.js
    trend_dates    = sorted(set(r[0] for r in trend_rows))
    trend_approved = []
    trend_rejected = []
    for d in trend_dates:
        a = next((r[2] for r in trend_rows if r[0] == d and r[1] == "Approved"), 0)
        r = next((r[2] for r in trend_rows if r[0] == d and r[1] == "Rejected"), 0)
        trend_approved.append(a)
        trend_rejected.append(r)

    # Feature importance from model
    importances = model.feature_importances_
    fi_pairs    = sorted(zip(FEATURES, importances.tolist()), key=lambda x: x[1], reverse=True)[:10]
    fi_labels   = [f[0] for f in fi_pairs]
    fi_values   = [round(f[1] * 100, 2) for f in fi_pairs]

    return render_template("dashboard.html",
        approved=approved,
        rejected=rejected,
        avg_prob=avg_prob,
        avg_loan=avg_loan,
        fi_labels=json.dumps(fi_labels),
        fi_values=json.dumps(fi_values),
        trend_dates=json.dumps(trend_dates),
        trend_approved=json.dumps(trend_approved),
        trend_rejected=json.dumps(trend_rejected)
    )


@app.route("/calculator")
def calculator():
    """EMI calculator page."""
    return render_template("calculator.html")


@app.route("/download-report")
def download_report():
    """Generate and download a PDF report for the last prediction."""
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
    sub_s   = ParagraphStyle('sub',   fontSize=11, fontName='Helvetica',
                              textColor=colors.HexColor('#5A6A7A'), spaceAfter=20)
    head_s  = ParagraphStyle('head',  fontSize=13, fontName='Helvetica-Bold',
                              textColor=colors.HexColor('#0D1B2A'), spaceBefore=14, spaceAfter=6)

    result_color = colors.HexColor('#27ae60') if data['result'] == 'Approved' else colors.HexColor('#e74c3c')

    elements = [
        Paragraph("LoanIQ — Loan Decision Report", title_s),
        Paragraph(f"Generated: {data['timestamp']}", sub_s),
        Spacer(1, 0.3*cm),

        Paragraph("Decision Summary", head_s),
        Table([
            ["Decision",         data['result']],
            ["Confidence Score", f"{data['probability']}%"],
            ["Risk Score",       f"{data.get('risk_score', 'N/A')} / 100"],
            ["Max Suggested Loan", f"{data.get('max_loan', 'N/A')}k"],
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