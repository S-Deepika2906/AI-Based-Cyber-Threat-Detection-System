import pandas as pd
import joblib


INPUT_FILE = "dataset/balanced_cicids2017.parquet"

print("Loading dataset...")

df = pd.read_parquet(INPUT_FILE)


# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]


# Convert Protocol
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


# Load model
print("Loading Isolation Forest...")

model = joblib.load(
    "models/isolation_forest.pkl"
)


# Get anomaly scores
scores = model.decision_function(X)


# Create result dataframe
result = pd.DataFrame({
    "Label": y.values,
    "AnomalyScore": scores
})


print("\n==============================")
print("ANOMALY SCORE ANALYSIS")
print("==============================")


for label in result["Label"].unique():

    class_scores = result[
        result["Label"] == label
    ]["AnomalyScore"]

    print(f"\n{label}")

    print("Count:", len(class_scores))

    print("Minimum:",
          round(class_scores.min(), 4))

    print("Maximum:",
          round(class_scores.max(), 4))

    print("Mean:",
          round(class_scores.mean(), 4))

    print("Median:",
          round(class_scores.median(), 4))