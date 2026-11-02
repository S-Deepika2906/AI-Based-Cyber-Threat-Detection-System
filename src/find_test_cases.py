import pandas as pd
import joblib
from pathlib import Path

# ==============================
# PATHS
# ==============================

DATASET_PATH = "dataset/balanced_cicids2017.parquet"
XGBOOST_MODEL_PATH = "models/xgboost_model.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
ISOLATION_MODEL_PATH = "models/isolation_forest.pkl"


# ==============================
# LOAD DATA
# ==============================

print("Loading dataset...")

df = pd.read_parquet(DATASET_PATH)

print("Dataset shape:", df.shape)

# Separate features and target
X = df.drop(columns=["Label"])
y = df["Label"]


# ==============================
# LOAD MODELS
# ==============================

print("\nLoading XGBoost model...")
xgb_model = joblib.load(XGBOOST_MODEL_PATH)

print("Loading label encoder...")
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("Loading Isolation Forest...")
isolation_model = joblib.load(ISOLATION_MODEL_PATH)


# ==============================
# XGBOOST PREDICTION
# ==============================

print("\nRunning XGBoost predictions...")

predicted_encoded = xgb_model.predict(X)

predicted_labels = label_encoder.inverse_transform(
    predicted_encoded.astype(int)
)

probabilities = xgb_model.predict_proba(X)

confidence = probabilities.max(axis=1)


# ==============================
# ISOLATION FOREST
# ==============================

print("Running Isolation Forest...")

anomaly_prediction = isolation_model.predict(X)

# -1 = anomaly
#  1 = normal

anomaly_status = [
    "Anomaly" if value == -1 else "Normal"
    for value in anomaly_prediction
]


# ==============================
# CREATE RESULTS
# ==============================

results = pd.DataFrame({
    "Record_Number": range(1, len(df) + 1),
    "Actual": y.values,
    "Prediction": predicted_labels,
    "Confidence": confidence,
    "Anomaly_Status": anomaly_status
})


# ==============================
# HYBRID DECISION
# ==============================

CONFIDENCE_THRESHOLD = 0.90

decisions = []

for _, row in results.iterrows():

    prediction = row["Prediction"]
    conf = row["Confidence"]
    anomaly = row["Anomaly_Status"]

    # Case 1:
    # Known attack + high confidence
    if prediction != "Benign" and conf >= CONFIDENCE_THRESHOLD:
        decision = f"Known Threat: {prediction}"

    # Case 2:
    # Benign + normal
    elif prediction == "Benign" and anomaly == "Normal":
        decision = "Benign"

    # Case 3:
    # Benign + anomaly
    elif prediction == "Benign" and anomaly == "Anomaly":
        decision = "Suspicious / Potential Novel Threat"

    # Case 4:
    # Low confidence + anomaly
    elif conf < CONFIDENCE_THRESHOLD and anomaly == "Anomaly":
        decision = "Suspicious / Potential Novel Threat"

    # Other low-confidence cases
    else:
        decision = "Needs Further Analysis"

    decisions.append(decision)


results["Final_Decision"] = decisions


# ==============================
# FIND TEST CASES
# ==============================

print("\n")
print("=" * 60)
print("HYBRID TEST CASES")
print("=" * 60)


# CASE 1
case1 = results[
    (results["Prediction"] != "Benign") &
    (results["Confidence"] >= CONFIDENCE_THRESHOLD)
]

print("\nCASE 1")
print("Known attack + High confidence")

if len(case1) > 0:
    print(case1.head(5).to_string(index=False))
else:
    print("No suitable record found.")


# CASE 2
case2 = results[
    (results["Prediction"] == "Benign") &
    (results["Anomaly_Status"] == "Normal")
]

print("\nCASE 2")
print("Benign + Normal")

if len(case2) > 0:
    print(case2.head(5).to_string(index=False))
else:
    print("No suitable record found.")


# CASE 3
case3 = results[
    (results["Prediction"] == "Benign") &
    (results["Anomaly_Status"] == "Anomaly")
]

print("\nCASE 3")
print("Benign + Anomaly")

if len(case3) > 0:
    print(case3.head(5).to_string(index=False))
else:
    print("No suitable record found.")


# CASE 4
case4 = results[
    (results["Confidence"] < CONFIDENCE_THRESHOLD) &
    (results["Anomaly_Status"] == "Anomaly")
]

print("\nCASE 4")
print("Low confidence + Anomaly")

if len(case4) > 0:
    print(case4.head(5).to_string(index=False))
else:
    print("No suitable record found.")


# ==============================
# SAVE RESULTS
# ==============================

results.to_csv(
    "dataset/hybrid_test_results.csv",
    index=False
)

print("\n")
print("=" * 60)
print("Results saved:")
print("dataset/hybrid_test_results.csv")
print("=" * 60)