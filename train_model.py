import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report
from xgboost import XGBClassifier

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
data['TotalIncome']       = data['ApplicantIncome'] + data['CoapplicantIncome']
data['EMI']               = data['LoanAmount'] / data['Loan_Amount_Term']
data['IncomePerDependent']= data['TotalIncome'] / (data['Dependents'] + 1)
data['LoanIncomeRatio']   = data['LoanAmount'] / (data['TotalIncome'] + 1)

# =========================
# Features & Target
# =========================
FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area',
    'TotalIncome', 'EMI', 'IncomePerDependent', 'LoanIncomeRatio'
]

X = data[FEATURES]
y = data['Loan_Status']

# =========================
# Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# XGBoost Model
# =========================
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("=" * 40)
print(f"  Accuracy : {acc * 100:.2f}%")
print(f"  Recall   : {recall:.4f}")
print("=" * 40)
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# =========================
# Save model + feature list
# =========================
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/loan_model.pkl", "wb"))
pickle.dump(FEATURES, open("model/features.pkl", "wb"))

print("Model saved → model/loan_model.pkl")
print("Features saved → model/features.pkl")