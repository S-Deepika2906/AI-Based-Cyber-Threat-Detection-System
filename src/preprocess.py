import pandas as pd
import os

DATASET_PATH = "dataset"

files = [
    "Benign-Monday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet"
]

print("Loading datasets...")

dataframes = []

for file in files:
    path = os.path.join(DATASET_PATH, file)

    print(f"Loading: {file}")

    df = pd.read_parquet(path)

    dataframes.append(df)

    print(f"Rows: {len(df)}")
    print(f"Labels:\n{df['Label'].value_counts()}\n")


print("Combining datasets...")

data = pd.concat(dataframes, ignore_index=True)

print("Combined shape:", data.shape)

print("\nOriginal label distribution:")
print(data["Label"].value_counts())


# Remove duplicate records
data = data.drop_duplicates()

print("\nAfter removing duplicates:", data.shape)


# Replace infinity values
data = data.replace([float("inf"), float("-inf")], pd.NA)

# Remove rows containing missing values
data = data.dropna()

print("After removing missing values:", data.shape)


# Save cleaned dataset
output_path = "dataset/cleaned_cicids2017.parquet"

data.to_parquet(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print(output_path)

print("\nFinal label distribution:")
print(data["Label"].value_counts())