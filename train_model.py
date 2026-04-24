import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

# =========================
# Load dataset
# =========================
data = pd.read_csv("dataset/train_u6lujuX_CVtuZ9i.csv")

# =========================
# 1. Separate columns by type (IMPORTANT FIX)
# =========================
num_cols = data.select_dtypes(include=[np.number]).columns
cat_cols = data.select_dtypes(include=['object']).columns

# =========================
# 2. Handle missing values safely
# =========================

# numeric → median
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

# categorical → mode
for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# =========================
# 3. Fix Dependents separately
# =========================
data['Dependents'] = data['Dependents'].replace('3+', 3)
data['Dependents'] = pd.to_numeric(data['Dependents'], errors='coerce')
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].median())
data['Dependents'] = data['Dependents'].astype(int)

# =========================
# 4. Encoding
# =========================
data['Gender'] = data['Gender'].map({'Male':1,'Female':0})
data['Married'] = data['Married'].map({'Yes':1,'No':0})
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0})
data['Property_Area'] = data['Property_Area'].map({'Urban':2,'Semiurban':1,'Rural':0})
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0})

# =========================
# 5. Final safety cleanup
# =========================
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(data.median(numeric_only=True))

# =========================
# Features & Target
# =========================
X = data.drop(['Loan_ID','Loan_Status'], axis=1)
y = data['Loan_Status']

# =========================
# Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Model
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# =========================
# Save model
# =========================
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/loan_model.pkl", "wb"))

print("Model saved successfully 🚀")