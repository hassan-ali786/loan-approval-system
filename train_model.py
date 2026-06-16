import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =========================
# Load dataset
# =========================
data = pd.read_csv("dataset/train.csv")

# =========================
# Handle missing values
# =========================
num_cols = data.select_dtypes(include=[np.number]).columns
cat_cols = data.select_dtypes(include=['object']).columns

data[num_cols] = data[num_cols].fillna(data[num_cols].median())
for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# =========================
# Fix Dependents
# =========================
data['Dependents'] = data['Dependents'].replace('3+', 3)
data['Dependents'] = pd.to_numeric(data['Dependents'], errors='coerce').fillna(0).astype(int)

# =========================
# Encoding
# =========================
data['Gender']        = data['Gender'].map({'Male': 1, 'Female': 0})
data['Married']       = data['Married'].map({'Yes': 1, 'No': 0})
data['Education']     = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
data['Loan_Status']   = data['Loan_Status'].map({'Y': 1, 'N': 0})

# =========================
# Feature Engineering
# =========================
data['TotalIncome']        = data['ApplicantIncome'] + data['CoapplicantIncome']
data['EMI']                = data['LoanAmount'] / data['Loan_Amount_Term']
data['IncomePerDependent'] = data['TotalIncome'] / (data['Dependents'] + 1)
data['LoanIncomeRatio']    = data['LoanAmount'] / (data['TotalIncome'] + 1)

FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area',
    'TotalIncome', 'EMI', 'IncomePerDependent', 'LoanIncomeRatio'
]

X = data[FEATURES]
y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Model Comparison
# =========================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42),
    "XGBoost":       XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                   random_state=42, eval_metric='logloss', verbosity=0),
    "LightGBM":      LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    class_weight='balanced', random_state=42, verbose=-1)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_name  = ""
best_model = None

print("=" * 50)
print(f"{'Model':<16} {'CV Accuracy':>12} {'Test Acc':>10} {'Recall':>8}")
print("-" * 50)

for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    recall   = recall_score(y_test, y_pred)
    cv_mean  = cv_scores.mean()

    print(f"{name:<16} {cv_mean*100:>10.2f}%  {test_acc*100:>8.2f}%  {recall:>6.4f}")

    if cv_mean > best_score:
        best_score = cv_mean
        best_name  = name
        best_model = model

print("=" * 50)
print(f"\n✓ Best Model: {best_name} (CV Accuracy: {best_score*100:.2f}%)\n")
print(classification_report(y_test, best_model.predict(X_test), target_names=["Rejected", "Approved"]))

# =========================
# Save best model
# =========================
os.makedirs("model", exist_ok=True)
pickle.dump(best_model, open("model/loan_model.pkl", "wb"))
pickle.dump(FEATURES,   open("model/features.pkl",   "wb"))

print(f"Saved: model/loan_model.pkl ({best_name})")