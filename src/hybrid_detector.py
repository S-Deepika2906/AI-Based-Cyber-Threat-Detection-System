import pandas as pd
import joblib
import numpy as np


# ==============================
# Load models
# ==============================

xgb_model = joblib.load(
    "models/xgboost_model.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

isolation_model = joblib.load(
    "models/isolation_forest.pkl"
)


# ==============================
# Load test dataset
# ==============================

df = pd.read_parquet(
    "dataset/balanced_cicids2017.parquet"
)


# ==============================
# Prepare features
# ==============================

X = df.drop(columns=["Label"])
y = df["Label"]


if X["Protocol"].dtype == "object":
    X["Protocol"] = X["Protocol"].astype("category").cat.codes


X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [float("inf"), float("-inf")],
    pd.NA
)


valid_rows = X.notna().all(axis=1)

X = X[valid_rows]
y = y[valid_rows]


# ==============================
# XGBoost prediction
# ==============================

xgb_predictions = xgb_model.predict(X)

xgb_probabilities = xgb_model.predict_proba(X)

predicted_labels = label_encoder.inverse_transform(
    xgb_predictions.astype(int)
)


# ==============================
# Isolation Forest
# ==============================

anomaly_scores = isolation_model.decision_function(X)

anomaly_predictions = isolation_model.predict(X)


# ==============================
# Hybrid decision
# ==============================

results = []

for i in range(len(X)):

    predicted_class = predicted_labels[i]

    confidence = np.max(
        xgb_probabilities[i]
    )

    anomaly_score = anomaly_scores[i]

    is_anomaly = (
        anomaly_predictions[i] == -1
    )

    # Hybrid decision
    if predicted_class == "Benign" and is_anomaly:

        final_decision = "Suspicious / Potential Novel Threat"

    elif predicted_class != "Benign":

        final_decision = (
            f"Known Threat: {predicted_class}"
        )

    else:

        final_decision = "Benign"

    results.append([
        y.iloc[i],
        predicted_class,
        round(float(confidence), 4),
        round(float(anomaly_score), 4),
        final_decision
    ])


# ==============================
# Create result dataframe
# ==============================

result_df = pd.DataFrame(
    results,
    columns=[
        "Actual",
        "XGBoost_Prediction",
        "XGBoost_Confidence",
        "Anomaly_Score",
        "Final_Decision"
    ]
)


print("\n==============================")
print("HYBRID DETECTION RESULTS")
print("==============================")

print(
    result_df["Final_Decision"].value_counts()
)


print("\nSample results:")

print(
    result_df.head(20).to_string(index=False)
)