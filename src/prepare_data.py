import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

INPUT_FILE = "dataset/balanced_cicids2017.parquet"

print("Loading balanced dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Dataset shape:", df.shape)

# Separate features and target
X = df.drop(columns=["Label"])
y = df["Label"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)

# Convert protocol to numeric if it is categorical
if X["Protocol"].dtype == "object":
    X["Protocol"] = X["Protocol"].astype("category").cat.codes

# Convert all remaining non-numeric columns
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values
X = X.replace([float("inf"), float("-inf")], pd.NA)

# Remove rows containing missing values
valid_rows = X.notna().all(axis=1)

X = X[valid_rows]
y = y[valid_rows]

# Encode attack labels
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nLabel mapping:")

for number, label in enumerate(label_encoder.classes_):
    print(number, "=", label)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nData preparation completed successfully!")