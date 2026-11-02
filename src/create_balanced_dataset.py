import pandas as pd

INPUT_FILE = "dataset/cleaned_cicids2017.parquet"
OUTPUT_FILE = "dataset/balanced_cicids2017.parquet"

# Number of samples we want from each class
SAMPLE_SIZE = {
    "Benign": 20000,
    "FTP-Patator": 5000,
    "SSH-Patator": 3000,
    "PortScan": 2000,
    "Bot": 1400
}

print("Loading cleaned dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Original shape:", df.shape)

balanced_parts = []

for label, size in SAMPLE_SIZE.items():

    class_data = df[df["Label"] == label]

    print(f"\n{label}")
    print("Available:", len(class_data))
    print("Selecting:", size)

    if len(class_data) < size:
        print("Warning: Not enough samples!")
        selected = class_data
    else:
        selected = class_data.sample(
            n=size,
            random_state=42
        )

    balanced_parts.append(selected)


balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

# Shuffle the final dataset
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nFinal balanced dataset:")
print(balanced_df["Label"].value_counts())

print("\nFinal shape:")
print(balanced_df.shape)

balanced_df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("\nBalanced dataset saved successfully!")
print(OUTPUT_FILE)