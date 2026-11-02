import pandas as pd
import joblib
import shap
import numpy as np


# ==============================
# 1. Load dataset
# ==============================

INPUT_FILE = "dataset/balanced_cicids2017.parquet"

print("Loading dataset...")

df = pd.read_parquet(INPUT_FILE)

# Take a small sample for explanation
df_sample = df.sample(
    n=10,
    random_state=42
).reset_index(drop=True)


# ==============================
# 2. Prepare features
# ==============================

X = df_sample.drop(columns=["Label"])


# Convert Protocol
if X["Protocol"].dtype == "object":
    X["Protocol"] = X["Protocol"].astype("category").cat.codes


# Convert all columns to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# Replace infinity
X = X.replace(
    [float("inf"), float("-inf")],
    pd.NA
)


# Remove invalid rows
valid_rows = X.notna().all(axis=1)

X = X[valid_rows]

df_sample = df_sample.loc[
    valid_rows
].reset_index(drop=True)


# ==============================
# 3. Load trained XGBoost model
# ==============================

print("Loading XGBoost model...")

model = joblib.load(
    "models/xgboost_model.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)


# ==============================
# 4. Create SHAP explainer
# ==============================

print("Creating SHAP explainer...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)


print("SHAP values generated!")


# ==============================
# 5. Get model predictions
# ==============================

predictions = model.predict(X)

predicted_labels = label_encoder.inverse_transform(
    predictions.astype(int)
)


# ==============================
# 6. Display explanations
# ==============================

print("\n==============================")
print("XAI EXPLANATIONS")
print("==============================")


for i in range(len(X)):

    predicted_class = predicted_labels[i]

    print("\n--------------------------------")
    print(f"Sample {i + 1}")
    print(f"Actual Label    : {df_sample.iloc[i]['Label']}")
    print(f"Prediction      : {predicted_class}")

    # Get SHAP values for predicted class
    sample_shap = shap_values[i]

    if isinstance(sample_shap, list):
        class_index = int(predictions[i])
        values = sample_shap[class_index]
    else:
        values = sample_shap[:, int(predictions[i])]

    # Find top 5 contributing features
    top_indices = np.argsort(
        np.abs(values)
    )[::-1][:5]

    print("\nTop contributing features:")

    for index in top_indices:

        feature_name = X.columns[index]

        feature_value = X.iloc[i, index]

        shap_value = values[index]

        print(
            f"{feature_name}: "
            f"value={feature_value:.4f}, "
            f"SHAP={shap_value:.4f}"
        )