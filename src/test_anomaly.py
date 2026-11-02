import pandas as pd
import joblib

from sklearn.metrics import classification_report


# ==============================
# 1. Load dataset
# ==============================

INPUT_FILE = "dataset/balanced_cicids2017.parquet"

print("Loading dataset...")

df = pd.read_parquet(INPUT_FILE)


# ==============================
# 2. Separate features and labels
# ==============================

X = df.drop(columns=["Label"])
y = df["Label"]


# ==============================
# 3. Convert Protocol
# ==============================

if X["Protocol"].dtype == "object":
    X["Protocol"] = X["Protocol"].astype("category").cat.codes


# Convert to numeric
X = X.apply(pd.to_numeric, errors="coerce")


# Replace infinity
X = X.replace(
    [float("inf"), float("-inf")],
    pd.NA
)


# Remove invalid rows
valid_rows = X.notna().all(axis=1)

X = X[valid_rows]
y = y[valid_rows]


# ==============================
# 4. Load Isolation Forest
# ==============================

print("\nLoading Isolation Forest...")

model = joblib.load(
    "models/isolation_forest.pkl"
)


# ==============================
# 5. Predict anomaly
# ==============================

print("\nDetecting anomalies...")

predictions = model.predict(X)


# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

y_pred = []

for prediction in predictions:

    if prediction == 1:
        y_pred.append("Benign")
    else:
        y_pred.append("Anomaly")


# ==============================
# 6. Display results
# ==============================

result = pd.DataFrame({
    "Actual": y.values,
    "Predicted": y_pred
})


print("\nAnomaly Detection Results:")
print(
    result["Predicted"].value_counts()
)


print("\nActual Class Distribution:")
print(
    result["Actual"].value_counts()
)


# ==============================
# 7. Compare attack detection
# ==============================

result["Actual_Anomaly"] = (
    result["Actual"] != "Benign"
)

result["Detected_Anomaly"] = (
    result["Predicted"] == "Anomaly"
)


print("\nAttack Detection Summary:")

for label in result["Actual"].unique():

    if label != "Benign":

        total = (
            result["Actual"] == label
        ).sum()

        detected = (
            (result["Actual"] == label) &
            (result["Predicted"] == "Anomaly")
        ).sum()

        detection_rate = (
            detected / total * 100
        )

        print(
            f"{label}: "
            f"{detected}/{total} "
            f"({detection_rate:.2f}%)"
        )