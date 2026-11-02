import pandas as pd
import joblib
import shap
import numpy as np


# ==============================
# Load dataset
# ==============================

df = pd.read_parquet(
    "dataset/balanced_cicids2017.parquet"
)


# Select one sample
sample = df.sample(
    n=1,
    random_state=100
).reset_index(drop=True)


actual_label = sample["Label"].iloc[0]

X = sample.drop(columns=["Label"])


# ==============================
# Prepare features
# ==============================

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

X = X.dropna(axis=1)


# ==============================
# Load models
# ==============================

model = joblib.load(
    "models/xgboost_model.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)


# ==============================
# Prediction
# ==============================

prediction = model.predict(X)[0]

predicted_label = label_encoder.inverse_transform(
    [int(prediction)]
)[0]


probabilities = model.predict_proba(X)[0]

confidence = probabilities[int(prediction)]


# ==============================
# SHAP
# ==============================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)


# ==============================
# Get SHAP values
# ==============================

if isinstance(shap_values, list):

    values = shap_values[int(prediction)][0]

else:

    values = shap_values[0, :, int(prediction)]


# ==============================
# Top features
# ==============================

top_indices = np.argsort(
    np.abs(values)
)[::-1][:5]


print("\n==============================")
print("AI-POWERED THREAT EXPLANATION")
print("==============================")

print(f"\nActual Label : {actual_label}")
print(f"Prediction   : {predicted_label}")
print(f"Confidence   : {confidence:.2%}")


print("\nTop contributing features:")

for rank, index in enumerate(top_indices, start=1):

    feature = X.columns[index]

    value = X.iloc[0, index]

    shap_value = values[index]

    direction = (
        "supporting the prediction"
        if shap_value > 0
        else "opposing the prediction"
    )

    print(
        f"{rank}. {feature}"
    )

    print(
        f"   Value: {value}"
    )

    print(
        f"   SHAP: {shap_value:.4f}"
    )

    print(
        f"   Effect: {direction}"
    )


# ==============================
# Human-readable summary
# ==============================

top_features = [
    X.columns[index]
    for index in top_indices[:3]
]

feature_text = ", ".join(top_features)


print("\n--------------------------------")
print("EXPLANATION")
print("--------------------------------")

print(
    f"The model classified this traffic as "
    f"'{predicted_label}' with "
    f"{confidence:.2%} confidence. "
    f"The prediction was mainly influenced by "
    f"{feature_text}."
)