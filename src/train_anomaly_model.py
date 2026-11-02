import pandas as pd
import os
import joblib

from sklearn.ensemble import IsolationForest


# ==============================
# 1. Load balanced dataset
# ==============================

INPUT_FILE = "dataset/balanced_cicids2017.parquet"

print("Loading dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Dataset shape:", df.shape)


# ==============================
# 2. Select only normal traffic
# ==============================

normal_data = df[df["Label"] == "Benign"].copy()

print("\nNormal traffic samples:", len(normal_data))


# ==============================
# 3. Separate features
# ==============================

X_normal = normal_data.drop(columns=["Label"])


# ==============================
# 4. Convert Protocol
# ==============================

if X_normal["Protocol"].dtype == "object":
    X_normal["Protocol"] = X_normal["Protocol"].astype("category").cat.codes


# Convert everything to numeric
X_normal = X_normal.apply(pd.to_numeric, errors="coerce")


# Replace infinite values
X_normal = X_normal.replace(
    [float("inf"), float("-inf")],
    pd.NA
)


# Remove invalid rows
X_normal = X_normal.dropna()


print("Final training samples:", len(X_normal))
print("Number of features:", X_normal.shape[1])


# ==============================
# 5. Train Isolation Forest
# ==============================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

model.fit(X_normal)

print("Isolation Forest training completed!")


# ==============================
# 6. Save model
# ==============================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/isolation_forest.pkl"
)

print("\nModel saved successfully:")
print("models/isolation_forest.pkl")