import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

DATASET_REPO_ID = "BharatBoddu/tourism-purchase-dataset"
DATASET_REPO_TYPE = "dataset"
INPUT_DATA_PATH = f"hf://datasets/{DATASET_REPO_ID}/tourism.csv"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset from Hugging Face
raw_df = pd.read_csv(INPUT_DATA_PATH)
print("Dataset loaded from Hugging Face.")

# Drop CustomerID — not a predictive feature
clean_df = raw_df.drop(columns=["CustomerID"], errors="ignore")

# Fill missing values
for col in clean_df.columns:
    if clean_df[col].dtype == "object":
        clean_df[col] = clean_df[col].fillna(clean_df[col].mode().iloc[0])
    else:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

print(f"Shape after cleaning: {clean_df.shape}")

# Train / test split (stratified on target)
target_col = "ProdTaken"
X = clean_df.drop(columns=[target_col])
y = clean_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Upload split datasets back to Hugging Face
for file_name in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name,
        repo_id=DATASET_REPO_ID,
        repo_type=DATASET_REPO_TYPE,
    )

print("Train/test splits uploaded to Hugging Face Dataset Hub.")
