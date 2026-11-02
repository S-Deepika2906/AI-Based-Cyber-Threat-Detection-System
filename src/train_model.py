import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ==============================
# 1. Load balanced dataset
# ==============================

INPUT_FILE = "dataset/balanced_cicids2017.parquet"

print("Loading dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Dataset shape:", df.shape)


# ==============================
# 2. Separate features and label
# ==============================

X = df.drop(columns=["Label"])
y = df["Label"]


# ==============================
# 3. Convert categorical data
# ==============================

if X["Protocol"].dtype == "object":
    X["Protocol"] = X["Protocol"].astype("category").cat.codes


# Convert remaining columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")


# Replace infinite values
X = X.replace([float("inf"), float("-inf")], pd.NA)


# Remove invalid rows
valid_rows = X.notna().all(axis=1)

X = X[valid_rows]
y = y[valid_rows]


# ==============================
# 4. Encode target labels
# ==============================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nClass mapping:")

for number, label in enumerate(label_encoder.classes_):
    print(f"{number} = {label}")


# ==============================
# 5. Train-test split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# 6. Create XGBoost model
# ==============================

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)


# ==============================
# 7. Train
# ==============================

model.fit(X_train, y_train)

print("Training completed!")


# ==============================
# 8. Prediction
# ==============================

y_pred = model.predict(X_test)


# ==============================
# 9. Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==============================
# 10. Save model
# ==============================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgboost_model.pkl")

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

print("\nModel saved:")
print("models/xgboost_model.pkl")

print("Label encoder saved:")
print("models/label_encoder.pkl")