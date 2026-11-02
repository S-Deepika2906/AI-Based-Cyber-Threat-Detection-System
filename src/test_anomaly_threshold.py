import pandas as pd
import joblib
import numpy as np

INPUT_FILE = "dataset/balanced_cicids2017.parquet"

print("Loading dataset...")

df = pd.read_parquet(INPUT_FILE)

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Convert Protocol
if X["Protocol"].dtype == "object":
    X["Protocol"] = X["Protocol"].astype("category").cat.codes

# Convert all columns to numeric
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

# Load Isolation Forest
model = joblib.load(
    "models/isolation_forest.pkl"
)

# Get anomaly scores
scores = model.decision_function(X)

result = pd.DataFrame({
    "Label": y.values,
    "Score": scores
})

print("\n==============================")
print("NORMAL BASELINE")
print("==============================")

benign_scores = result[
    result["Label"] == "Benign"
]["Score"]

print("Benign samples:", len(benign_scores))

print("Benign mean:",
      round(benign_scores.mean(), 4))

print("Benign standard deviation:",
      round(benign_scores.std(), 4))

# Calculate lower percentiles
for percentile in [1, 2, 5, 10]:

    threshold = np.percentile(
        benign_scores,
        percentile
    )

    print(
        f"Benign {percentile}th percentile:",
        round(threshold, 4)
    )


print("\n==============================")
print("ATTACK SCORE DISTRIBUTION")
print("==============================")

for label in result["Label"].unique():

    if label == "Benign":
        continue

    attack_scores = result[
        result["Label"] == label
    ]["Score"]

    print(
        f"{label}:",
        "mean =", round(attack_scores.mean(), 4),
        "| 5th percentile =",
        round(np.percentile(attack_scores, 5), 4)
    )