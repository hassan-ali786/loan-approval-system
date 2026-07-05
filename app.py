from flask import Flask, request, render_template, jsonify, make_response, session, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
import pickle
import numpy as np
import pandas as pd
import sqlite3
import shap
import lime
import lime.lime_tabular
import json
import csv
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from auth import (
    User, init_user_db, create_user, get_user_by_email, get_user_by_id,
    verify_password, get_all_users, toggle_block_user, delete_user, is_user_blocked,
    get_password_hash, update_password,
    init_reset_table, create_reset_token, get_reset_token, mark_token_used,
    init_notifications_table, add_notification, get_notifications,
    get_unread_count, mark_notifications_read,
    init_verification_table, create_verification_token, verify_email_token, is_user_verified,
    init_activity_log, log_activity, get_activity_log
)
from functools import wraps

app = Flask(__name__)
app.secret_key = "loaniq-secret-key-2024"

# ─────────────────────────────────────────
# Flask-Login setup
# ─────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view        = "welcome"
login_manager.login_message     = None

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(int(user_id))


def admin_required(f):
    """Restrict a route to logged-in admins only."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not getattr(current_user, 'is_admin', False):
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return wrapper


@app.context_processor
def inject_notifications():
    """Make unread notification count available in every template."""
    if current_user.is_authenticated and getattr(current_user, 'is_admin', False):
        return {"unread_count": get_unread_count()}
    return {"unread_count": 0}


@app.route("/admin/notifications")
@login_required
@admin_required
def admin_notifications():
    """Return notifications as JSON for the bell dropdown."""
    notifications = get_notifications(limit=20)
    mark_notifications_read()
    return jsonify([
        {"id": n[0], "message": n[1], "is_read": n[2], "created_at": n[3]}
        for n in notifications
    ])

# ─────────────────────────────────────────
# Flask-Mail setup (configure your SMTP)
# ─────────────────────────────────────────
import os

app.config['MAIL_SERVER']   = 'smtp.gmail.com'
app.config['MAIL_PORT']     = 587
app.config['MAIL_USE_TLS']  = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'your-email@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your-app-password')
app.config['MAIL_DEFAULT_SENDER'] = f"LoanIQ <{app.config['MAIL_USERNAME']}>"
mail = Mail(app)

# ─────────────────────────────────────────
# Load model & features
# ─────────────────────────────────────────
model    = pickle.load(open("model/loan_model.pkl", "rb"))
FEATURES = pickle.load(open("model/features.pkl", "rb"))
explainer = shap.TreeExplainer(model)

# LIME explainer — built from training data at startup
try:
    _train_df = pd.read_csv("dataset/train.csv")
    _train_df['Dependents']    = _train_df['Dependents'].replace('3+', 3)
    _train_df['Dependents']    = pd.to_numeric(_train_df['Dependents'], errors='coerce').fillna(0)
    _train_df['Gender']        = _train_df['Gender'].map({'Male': 1, 'Female': 0}).fillna(1)
    _train_df['Married']       = _train_df['Married'].map({'Yes': 1, 'No': 0}).fillna(1)
    _train_df['Education']     = _train_df['Education'].map({'Graduate': 1, 'Not Graduate': 0}).fillna(1)
    _train_df['Self_Employed'] = _train_df['Self_Employed'].map({'Yes': 1, 'No': 0}).fillna(0)
    _train_df['Property_Area'] = _train_df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0}).fillna(2)
    _train_df['TotalIncome']        = _train_df['ApplicantIncome'] + _train_df['CoapplicantIncome']
    _train_df['EMI']                = _train_df['LoanAmount'] / _train_df['Loan_Amount_Term'].replace(0, 1)
    _train_df['IncomePerDependent'] = _train_df['TotalIncome'] / (_train_df['Dependents'] + 1)
    _train_df['LoanIncomeRatio']    = _train_df['LoanAmount'] / (_train_df['TotalIncome'] + 1)
    _train_X = _train_df[FEATURES].fillna(0).values

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=_train_X,
        feature_names=FEATURES,
        class_names=["Rejected", "Approved"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )
except Exception as _e:
    lime_explainer = None
    print(f"LIME explainer init failed (dataset missing?): {_e}")

# ─────────────────────────────────────────
# DB init
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
            max_loan         REAL,
            user_id          INTEGER DEFAULT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()
init_user_db()
init_reset_table()
init_notifications_table()
init_verification_table()
init_activity_log()


# ─────────────────────────────────────────
# ML helpers
# ─────────────────────────────────────────
def get_shap_factors(X_input):
    """Extract top-5 SHAP factors safely for any tree model."""
    shap_values = explainer.shap_values(X_input)
    if isinstance(shap_values, list):
        sv_raw = np.array(shap_values[1]).flatten()
    else:
        sv_raw = np.array(shap_values).flatten()
    sv_floats = [float(v) for v in sv_raw[:len(FEATURES)]]
    pairs = sorted(zip(FEATURES, sv_floats), key=lambda x: abs(x[1]), reverse=True)[:5]
    return [{"feature": k, "value": round(v, 4), "abs_value": round(abs(v), 4), "width": min(round(abs(v) * 300, 1), 100)} for k, v in pairs]


def get_lime_factors(X_input_array, prediction_class):
    """
    Get top-5 LIME factors for a single prediction.
    Returns list of {feature, effect, direction} dicts.
    """
    if lime_explainer is None:
        return []
    try:
        exp = lime_explainer.explain_instance(
            X_input_array,
            model.predict_proba,
            num_features=5,
            top_labels=1
        )
        label = prediction_class  # 0=Rejected, 1=Approved
        lime_list = exp.as_list(label=label)
        result = []
        for feat_desc, effect in lime_list:
            result.append({
                "feature":   feat_desc,
                "effect":    round(effect, 4),
                "abs_effect": round(abs(effect), 4),
                "direction": "positive" if effect > 0 else "negative",
                "width":     min(round(abs(effect) * 500, 1), 100)
            })
        return result
    except Exception as e:
        print(f"LIME error: {e}")
        return []



    """Convert approval probability to 0-100 risk score."""
    return round(proba_approved * 100)


def suggest_max_loan(total_income, loan_term, credit_history):
    """Suggest max loan based on 40% EMI rule."""
    monthly_income = total_income / 12
    max_emi        = monthly_income * 0.40
    r              = 0.085 / 12
    n              = loan_term
    if r > 0 and n > 0:
        max_loan = max_emi * (1 - (1 + r) ** (-n)) / r
    else:
        max_loan = max_emi * n
    if credit_history == 0:
        max_loan *= 0.6
    return round(max_loan / 1000, 1)


def send_result_email(to_email, username, result, probability, risk_score, max_loan):
    """Send loan decision email notification."""
    try:
        subject = f"LoanIQ — Your Loan Application: {result}"
        color   = "#27ae60" if result == "Approved" else "#e74c3c"
        body    = f"""
        <div style="font-family:Arial,sans-serif; max-width:500px; margin:auto;">
            <h2 style="color:#0D1B2A;">LoanIQ — Loan Decision</h2>
            <p>Hi <strong>{username}</strong>,</p>
            <p>Your loan application has been processed.</p>
            <div style="background:#f9f9f9; border-left:4px solid {color}; padding:16px; border-radius:8px; margin:20px 0;">
                <h3 style="color:{color}; margin:0;">Decision: {result}</h3>
                <p style="margin:8px 0 0;">Confidence: <strong>{probability}%</strong></p>
                <p style="margin:4px 0 0;">Risk Score: <strong>{risk_score} / 100</strong></p>
                <p style="margin:4px 0 0;">Suggested Max Loan: <strong>{max_loan}k</strong></p>
            </div>
            <p style="color:#888; font-size:12px;">This is an automated message from LoanIQ.</p>
        </div>
        """
        msg      = Message(subject=subject, recipients=[to_email], html=body)
        mail.send(msg)
    except Exception as e:
        print(f"Email error: {e}")   # Non-blocking — app continues even if email fails


# ─────────────────────────────────────────
# Auth routes
# ─────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not username or not email or not password:
            return render_template("signup.html", error="All fields are required.")
        if len(password) < 6:
            return render_template("signup.html", error="Password must be at least 6 characters.")
        if create_user(username, email, password):
            add_notification(f"New user registered: {username} ({email})")
            # Send verification email
            row   = get_user_by_email(email)
            if row:
                token       = create_verification_token(row[0])
                verify_url  = url_for("verify_email", token=token, _external=True)
                try:
                    msg = Message(
                        subject="LoanIQ — Verify your email",
                        recipients=[email],
                        html=f"""
                        <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;">
                            <h2 style="color:#0D1B2A;">LoanIQ — Email Verification</h2>
                            <p>Hi <strong>{username}</strong>, welcome to LoanIQ!</p>
                            <p>Please verify your email address to unlock full access.</p>
                            <p><a href="{verify_url}" style="background:#B8902F;color:#0B0E14;
                               padding:10px 20px;border-radius:8px;text-decoration:none;
                               display:inline-block;">Verify Email</a></p>
                            <p style="color:#888;font-size:12px;">If you didn't create this account, ignore this email.</p>
                        </div>
                        """
                    )
                    mail.send(msg)
                except Exception as e:
                    print(f"Verification email error: {e}")
            return redirect(url_for("login", registered=1))
        return render_template("signup.html", error="Username or email already taken.")
    return render_template("signup.html")


@app.route("/verify-email/<token>")
def verify_email(token):
    """Verify a user's email address via one-time token."""
    user_id = verify_email_token(token)
    if user_id:
        return redirect(url_for("login", verified=1))
    return render_template("login.html", error="Verification link is invalid or already used.")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        row      = get_user_by_email(email)
        if row and verify_password(row[3], password):
            if row[5]:  # is_blocked
                return render_template("login.html", error="This account has been blocked. Contact an administrator.")
            user = User(row[0], row[1], row[2], row[4])  # includes is_admin
            login_user(user)
            log_activity(row[0], row[1], "login", "User signed in", request.remote_addr)
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid email or password.")
    registered = request.args.get("registered")
    return render_template("login.html", registered=registered)


@app.route("/logout")
@login_required
def logout():
    log_activity(current_user.id, current_user.username, "logout", "User signed out", request.remote_addr)
    logout_user()
    return redirect(url_for("login"))


@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    """Let any logged-in user (including admins) change their own password."""
    if request.method == "POST":
        current_pw = request.form.get("current_password", "")
        new_pw      = request.form.get("new_password", "")
        confirm_pw  = request.form.get("confirm_password", "")

        stored_hash = get_password_hash(current_user.id)
        if not stored_hash or not verify_password(stored_hash, current_pw):
            return render_template("change_password.html", error="Current password is incorrect.")
        if len(new_pw) < 6:
            return render_template("change_password.html", error="New password must be at least 6 characters.")
        if new_pw != confirm_pw:
            return render_template("change_password.html", error="New passwords do not match.")

        update_password(current_user.id, new_pw)
        return render_template("change_password.html", success="Password updated successfully.")

    return render_template("change_password.html")


@app.route("/profile")
@login_required
def profile():
    """User profile page — stats, account info, verification status."""
    if getattr(current_user, 'is_admin', False):
        return redirect(url_for("admin_panel"))

    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (current_user.id,))
    total_predictions = c.fetchone()[0]
    c.execute("SELECT result, COUNT(*) FROM predictions WHERE user_id = ? GROUP BY result", (current_user.id,))
    counts = dict(c.fetchall())
    c.execute("SELECT AVG(probability) FROM predictions WHERE user_id = ?", (current_user.id,))
    avg_prob = round(c.fetchone()[0] or 0, 1)
    c.execute("SELECT created_at, is_verified FROM users WHERE id = ?", (current_user.id,))
    user_row = c.fetchone()
    conn.close()

    return render_template("profile.html",
        total_predictions=total_predictions,
        approved=counts.get("Approved", 0),
        rejected=counts.get("Rejected", 0),
        avg_prob=avg_prob,
        joined=user_row[0] if user_row else "N/A",
        is_verified=bool(user_row[1]) if user_row else False
    )
def forgot_password():
    """Request a password reset link via email."""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        row   = get_user_by_email(email)
        if row:
            token = create_reset_token(row[0])
            reset_url = url_for("reset_password", token=token, _external=True)
            try:
                msg = Message(
                    subject="LoanIQ — Reset your password",
                    recipients=[email],
                    html=f"""
                    <div style="font-family:Arial,sans-serif; max-width:480px; margin:auto;">
                        <h2 style="color:#0D1B2A;">LoanIQ — Password Reset</h2>
                        <p>We received a request to reset your password.</p>
                        <p><a href="{reset_url}" style="background:#B8902F; color:#0B0E14; padding:10px 20px;
                           border-radius:8px; text-decoration:none; display:inline-block;">Reset Password</a></p>
                        <p style="color:#888; font-size:12px;">This link is valid for one use only. If you didn't request this, ignore this email.</p>
                    </div>
                    """
                )
                mail.send(msg)
            except Exception as e:
                print(f"Email error: {e}")
        # Always show the same message — don't reveal whether the email exists
        return render_template("forgot_password.html",
            success="If an account exists for that email, a reset link has been sent.")
    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    """Reset password using a valid one-time token."""
    token_data = get_reset_token(token)
    if not token_data or token_data[2] == 1:
        return render_template("reset_password.html", invalid=True)

    if request.method == "POST":
        new_pw     = request.form.get("new_password", "")
        confirm_pw = request.form.get("confirm_password", "")
        if len(new_pw) < 6:
            return render_template("reset_password.html", token=token, error="Password must be at least 6 characters.")
        if new_pw != confirm_pw:
            return render_template("reset_password.html", token=token, error="Passwords do not match.")

        update_password(token_data[0], new_pw)
        mark_token_used(token)
        return redirect(url_for("login", reset=1))

    return render_template("reset_password.html", token=token)


# ─────────────────────────────────────────
# Main routes
# ─────────────────────────────────────────
@app.route("/welcome")
def welcome():
    """Public landing page — shown to logged-out visitors."""
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    return render_template("landing.html")


@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    if getattr(current_user, 'is_admin', False):
        return redirect(url_for("admin_panel"))
    if request.method == "POST":
        try:
            applicant_income   = float(request.form.get('ApplicantIncome', 0))
            coapplicant_income = float(request.form.get('CoapplicantIncome', 0))
            loan_amount        = float(request.form.get('LoanAmount', 150))
            loan_term          = float(request.form.get('Loan_Amount_Term', 360))
            dependents         = int(request.form.get('Dependents', 0))
            credit_history     = int(request.form.get('Credit_History', 1))

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

            X_input        = pd.DataFrame([[features_dict[f] for f in FEATURES]], columns=FEATURES)
            prediction     = model.predict(X_input)[0]
            proba_array    = model.predict_proba(X_input)[0]
            proba_approved = float(proba_array[1])
            probability    = round(proba_approved * 100, 1) if prediction == 1 else round(float(proba_array[0]) * 100, 1)
            result         = "Approved" if prediction == 1 else "Rejected"
            risk_score     = compute_risk_score(proba_approved)
            max_loan       = suggest_max_loan(total_income, loan_term, credit_history)
            top_factors    = get_shap_factors(X_input)
            lime_factors   = get_lime_factors(X_input.values[0], int(prediction))

            # Persist to DB with user_id
            conn = sqlite3.connect("history.db")
            c    = conn.cursor()
            c.execute('''
                INSERT INTO predictions
                    (timestamp, result, probability, risk_score,
                     applicant_income, loan_amount, credit_history,
                     top_factors, max_loan, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                result, probability, risk_score,
                applicant_income, loan_amount, credit_history,
                json.dumps(top_factors), max_loan,
                current_user.id
            ))
            conn.commit()
            conn.close()

            # Session for PDF
            session['last_result'] = {
                'result': result, 'probability': probability,
                'risk_score': risk_score, 'max_loan': max_loan,
                'top_factors': top_factors, 'income': applicant_income,
                'co_income': coapplicant_income, 'loan_amount': loan_amount,
                'loan_term': loan_term, 'credit': credit_history,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            }

            # Send email notification (non-blocking)
            send_result_email(
                current_user.email, current_user.username,
                result, probability, risk_score, max_loan
            )

            # Log activity
            log_activity(current_user.id, current_user.username,
                         "prediction", f"Result: {result} ({probability}%)", request.remote_addr)

            return render_template("result.html",
                result=result, probability=probability,
                risk_score=risk_score, max_loan=max_loan,
                top_factors=top_factors,
                lime_factors=lime_factors
            )

        except Exception as e:
            print("ERROR:", e)
            return render_template("index.html", error="Something went wrong. Please check your inputs.")

    return render_template("index.html")


@app.route("/history")
@login_required
def history():
    """
    Regular users see only their own predictions.
    Admins see every prediction across all users, with the applicant's username shown.
    Supports ?filter=Approved/Rejected and ?page=N pagination.
    """
    filter_result = request.args.get('filter')
    is_admin_view = getattr(current_user, 'is_admin', False)
    page          = max(1, int(request.args.get('page', 1)))
    per_page      = 15

    conn = sqlite3.connect("history.db")
    c    = conn.cursor()

    if is_admin_view:
        base  = "FROM predictions p LEFT JOIN users u ON u.id = p.user_id"
        where = "WHERE p.result = ?" if filter_result in ("Approved","Rejected") else ""
        params = [filter_result] if filter_result in ("Approved","Rejected") else []

        c.execute(f"SELECT COUNT(*) {base} {where}", params)
        total = c.fetchone()[0]
        c.execute(f"SELECT p.*, u.username {base} {where} ORDER BY p.id DESC LIMIT ? OFFSET ?",
                  params + [per_page, (page-1)*per_page])
    else:
        base  = "FROM predictions WHERE user_id = ?"
        params = [current_user.id]
        if filter_result in ("Approved","Rejected"):
            base  += " AND result = ?"
            params.append(filter_result)

        c.execute(f"SELECT COUNT(*) {base}", params)
        total = c.fetchone()[0]
        c.execute(f"SELECT * {base} ORDER BY id DESC LIMIT ? OFFSET ?",
                  params + [per_page, (page-1)*per_page])

    rows        = c.fetchall()
    total_pages = max(1, (total + per_page - 1) // per_page)
    conn.close()

    return render_template("history.html",
        rows=rows,
        active_filter=filter_result,
        is_admin_view=is_admin_view,
        page=page,
        total_pages=total_pages,
        total=total
    )


@app.route("/export-csv")
@login_required
def export_csv():
    """Export prediction history as CSV — all users for admin, own history for regular users."""
    is_admin_view = getattr(current_user, 'is_admin', False)
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()

    if is_admin_view:
        c.execute("""
            SELECT p.id, p.timestamp, p.result, p.probability, p.risk_score,
                   p.applicant_income, p.loan_amount, p.credit_history, p.max_loan, u.username
            FROM predictions p LEFT JOIN users u ON u.id = p.user_id
            ORDER BY p.id DESC
        """)
        header = ["ID", "Timestamp", "Result", "Confidence (%)", "Risk Score",
                   "Applicant Income", "Loan Amount (k)", "Credit History", "Max Loan (k)", "Applicant"]
    else:
        c.execute("""
            SELECT id, timestamp, result, probability, risk_score,
                   applicant_income, loan_amount, credit_history, max_loan
            FROM predictions WHERE user_id = ? ORDER BY id DESC
        """, (current_user.id,))
        header = ["ID", "Timestamp", "Result", "Confidence (%)", "Risk Score",
                   "Applicant Income", "Loan Amount (k)", "Credit History", "Max Loan (k)"]

    rows = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=LoanIQ_History.csv"
    response.headers["Content-Type"]        = "text/csv"
    return response


@app.route("/dashboard")
@login_required
def dashboard():
    """Analytics dashboard — admins see system-wide data, regular users see only their own."""
    is_admin_view = getattr(current_user, 'is_admin', False)
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()

    if is_admin_view:
        c.execute("SELECT result, COUNT(*) FROM predictions GROUP BY result")
        counts = dict(c.fetchall())
        c.execute("SELECT AVG(probability) FROM predictions")
        avg_prob = round(c.fetchone()[0] or 0, 1)
        c.execute("SELECT AVG(loan_amount) FROM predictions")
        avg_loan = round(c.fetchone()[0] or 0, 1)
        c.execute("""
            SELECT DATE(timestamp) as day, result, COUNT(*) as cnt
            FROM predictions
            WHERE timestamp >= DATE('now', '-30 days')
            GROUP BY day, result ORDER BY day ASC
        """)
    else:
        c.execute("SELECT result, COUNT(*) FROM predictions WHERE user_id = ? GROUP BY result", (current_user.id,))
        counts = dict(c.fetchall())
        c.execute("SELECT AVG(probability) FROM predictions WHERE user_id = ?", (current_user.id,))
        avg_prob = round(c.fetchone()[0] or 0, 1)
        c.execute("SELECT AVG(loan_amount) FROM predictions WHERE user_id = ?", (current_user.id,))
        avg_loan = round(c.fetchone()[0] or 0, 1)
        c.execute("""
            SELECT DATE(timestamp) as day, result, COUNT(*) as cnt
            FROM predictions WHERE user_id = ?
            AND timestamp >= DATE('now', '-30 days')
            GROUP BY day, result ORDER BY day ASC
        """, (current_user.id,))

    trend_rows = c.fetchall()
    conn.close()

    approved = counts.get("Approved", 0)
    rejected = counts.get("Rejected", 0)

    trend_dates    = sorted(set(r[0] for r in trend_rows))
    trend_approved = [next((r[2] for r in trend_rows if r[0] == d and r[1] == "Approved"), 0) for d in trend_dates]
    trend_rejected = [next((r[2] for r in trend_rows if r[0] == d and r[1] == "Rejected"), 0) for d in trend_dates]

    importances = model.feature_importances_
    fi_pairs    = sorted(zip(FEATURES, importances.tolist()), key=lambda x: x[1], reverse=True)[:10]
    fi_labels   = [f[0] for f in fi_pairs]
    fi_values   = [round(f[1] * 100, 2) for f in fi_pairs]

    return render_template("dashboard.html",
        approved=approved, rejected=rejected,
        is_admin_view=is_admin_view,
        avg_prob=avg_prob, avg_loan=avg_loan,
        fi_labels=json.dumps(fi_labels), fi_values=json.dumps(fi_values),
        trend_dates=json.dumps(trend_dates),
        trend_approved=json.dumps(trend_approved),
        trend_rejected=json.dumps(trend_rejected)
    )


@app.route("/compare", methods=["GET", "POST"])
@login_required
def compare():
    """Loan comparison tool — compare up to 3 scenarios side by side."""
    if getattr(current_user, 'is_admin', False):
        return redirect(url_for("admin_panel"))
    results = []
    if request.method == "POST":
        scenarios = []
        for i in range(1, 4):
            try:
                ai   = float(request.form.get(f'ApplicantIncome_{i}', 0))
                ci   = float(request.form.get(f'CoapplicantIncome_{i}', 0))
                la   = float(request.form.get(f'LoanAmount_{i}', 150))
                lt   = float(request.form.get(f'Loan_Amount_Term_{i}', 360))
                dep  = int(request.form.get(f'Dependents_{i}', 0))
                cr   = int(request.form.get(f'Credit_History_{i}', 1))
                ti   = ai + ci
                emi  = la / lt if lt > 0 else 0
                fd   = {
                    'Gender': 1, 'Married': 1, 'Dependents': dep, 'Education': 1,
                    'Self_Employed': 0, 'ApplicantIncome': ai, 'CoapplicantIncome': ci,
                    'LoanAmount': la, 'Loan_Amount_Term': lt, 'Credit_History': cr,
                    'Property_Area': 2, 'TotalIncome': ti, 'EMI': emi,
                    'IncomePerDependent': ti / (dep + 1),
                    'LoanIncomeRatio': la / (ti + 1)
                }
                X   = pd.DataFrame([[fd[f] for f in FEATURES]], columns=FEATURES)
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0]
                pa   = float(prob[1])
                scenarios.append({
                    'label':       f"Scenario {i}",
                    'income':      ai,
                    'co_income':   ci,
                    'loan_amount': la,
                    'loan_term':   lt,
                    'credit':      "Good" if cr == 1 else "Poor",
                    'result':      "Approved" if pred == 1 else "Rejected",
                    'probability': round(pa * 100, 1) if pred == 1 else round(float(prob[0]) * 100, 1),
                    'risk_score':  compute_risk_score(pa),
                    'max_loan':    suggest_max_loan(ti, lt, cr),
                    'emi':         round(emi, 2)
                })
            except Exception:
                scenarios.append(None)
        results = [s for s in scenarios if s]
    return render_template("compare.html", results=results)


@app.route("/calculator")
@login_required
def calculator():
    if getattr(current_user, 'is_admin', False):
        return redirect(url_for("admin_panel"))
    return render_template("calculator.html")


# ─────────────────────────────────────────
# Admin panel routes
# ─────────────────────────────────────────
@app.route("/admin")
@login_required
@admin_required
def admin_panel():
    """Full admin overview — system stats + searchable, filterable, paginated user list."""
    search     = request.args.get('search', '').strip()
    status     = request.args.get('status', 'all')   # all | active | blocked
    page       = max(1, int(request.args.get('page', 1)))
    per_page   = 10

    conn = sqlite3.connect("history.db")
    c    = conn.cursor()

    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = c.fetchone()[0]

    c.execute("SELECT result, COUNT(*) FROM predictions GROUP BY result")
    counts = dict(c.fetchall())

    c.execute("SELECT AVG(probability) FROM predictions")
    avg_prob = round(c.fetchone()[0] or 0, 1)

    # Build filtered user query
    where_clauses = []
    params        = []
    if search:
        where_clauses.append("(u.username LIKE ? OR u.email LIKE ?)")
        params += [f'%{search}%', f'%{search}%']
    if status == 'active':
        where_clauses.append("u.is_blocked = 0")
    elif status == 'blocked':
        where_clauses.append("u.is_blocked = 1")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Total filtered count for pagination
    c.execute(f"SELECT COUNT(*) FROM users u {where_sql}", params)
    total_filtered = c.fetchone()[0]
    total_pages    = max(1, (total_filtered + per_page - 1) // per_page)
    offset         = (page - 1) * per_page

    c.execute(f'''
        SELECT u.id, u.username, u.email, u.is_admin, u.is_blocked, u.created_at,
               COUNT(p.id) as prediction_count
        FROM users u
        LEFT JOIN predictions p ON p.user_id = u.id
        {where_sql}
        GROUP BY u.id
        ORDER BY u.id ASC
        LIMIT ? OFFSET ?
    ''', params + [per_page, offset])
    users = c.fetchall()
    conn.close()

    return render_template("admin.html",
        total_users=total_users,
        total_predictions=total_predictions,
        approved=counts.get("Approved", 0),
        rejected=counts.get("Rejected", 0),
        avg_prob=avg_prob,
        users=users,
        search=search,
        status=status,
        page=page,
        total_pages=total_pages,
        total_filtered=total_filtered
    )


@app.route("/admin/toggle-block/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def admin_toggle_block(user_id):
    """Block or unblock a user. Admins cannot block themselves or other admins."""
    conn = sqlite3.connect("history.db")
    c    = conn.cursor()
    c.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()

    if row and row[0] == 1:
        flash("Admins cannot be blocked.")
        return redirect(url_for("admin_panel"))

    toggle_block_user(user_id)
    log_activity(current_user.id, current_user.username, "block",
                 f"Toggled block status for user_id={user_id}", request.remote_addr)
    return redirect(url_for("admin_panel"))


@app.route("/admin/delete/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def admin_delete_user(user_id):
    """Delete a non-admin user and their prediction history."""
    if user_id == current_user.id:
        flash("You cannot delete your own account.")
        return redirect(url_for("admin_panel"))

    deleted = delete_user(user_id)
    if deleted:
        log_activity(current_user.id, current_user.username, "delete",
                     f"Deleted user_id={user_id} and their predictions", request.remote_addr)
    else:
        flash("Could not delete this user — admins are protected.")
    return redirect(url_for("admin_panel"))


@app.route("/admin/note/<int:prediction_id>", methods=["POST"])
@login_required
@admin_required
def admin_add_note(prediction_id):
    """Admin adds or updates a note on a specific prediction."""
    note = request.form.get("note", "").strip()[:500]  # max 500 chars

    conn = sqlite3.connect("history.db")
    c    = conn.cursor()

    # Add officer_note column if it doesn't exist yet
    try:
        c.execute("ALTER TABLE predictions ADD COLUMN officer_note TEXT DEFAULT NULL")
    except Exception:
        pass

    c.execute("UPDATE predictions SET officer_note = ? WHERE id = ?", (note, prediction_id))
    conn.commit()
    conn.close()

    # Return to history, preserving page/filter params
    referrer = request.referrer or url_for("history")
    return redirect(referrer)


@app.route("/admin/retrain", methods=["POST"])
@login_required
@admin_required
def admin_retrain():
    """Retrain the model using the existing dataset. Reloads the model in memory after training."""
    global model, explainer
    try:
        import subprocess
        result = subprocess.run(
            ["python", "train_model.py"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            model    = pickle.load(open("model/loan_model.pkl", "rb"))
            explainer = shap.TreeExplainer(model)
            add_notification("Model retrained successfully by admin.")
            flash("Model retrained successfully.")
        else:
            flash(f"Retrain failed: {result.stderr[:200]}")
    except Exception as e:
        flash(f"Retrain error: {str(e)[:200]}")
    return redirect(url_for("admin_panel"))


@app.route("/admin/activity-log")
@login_required
@admin_required
def admin_activity_log():
    """Full activity log — all user actions."""
    logs = get_activity_log(limit=100)
    return render_template("activity_log.html", logs=logs)


@app.route("/admin/export-users-csv")
@login_required
@admin_required
def admin_export_users_csv():
    """Export the full user list as CSV."""
    users = get_all_users()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Username", "Email", "Is Admin", "Is Blocked", "Created At", "Prediction Count"])
    for u in users:
        writer.writerow(u)

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=LoanIQ_Users.csv"
    response.headers["Content-Type"]        = "text/csv"
    return response


@app.route("/download-report")
@login_required
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
    sub_s   = ParagraphStyle('sub', fontSize=11, fontName='Helvetica',
                              textColor=colors.HexColor('#5A6A7A'), spaceAfter=20)
    head_s  = ParagraphStyle('head', fontSize=13, fontName='Helvetica-Bold',
                              textColor=colors.HexColor('#0D1B2A'), spaceBefore=14, spaceAfter=6)
    result_color = colors.HexColor('#27ae60') if data['result'] == 'Approved' else colors.HexColor('#e74c3c')

    elements = [
        Paragraph("LoanIQ — Loan Decision Report", title_s),
        Paragraph(f"Generated: {data['timestamp']}  |  User: {getattr(current_user, 'username', 'N/A')}", sub_s),
        Spacer(1, 0.3*cm),
        Paragraph("Decision Summary", head_s),
        Table([
            ["Decision",           data['result']],
            ["Confidence Score",   f"{data['probability']}%"],
            ["Risk Score",         f"{data.get('risk_score', 'N/A')} / 100"],
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)