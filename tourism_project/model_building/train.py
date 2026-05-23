import os
import joblib
import pandas as pd
import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_REPO_ID = "BharatBoddu/tourism-purchase-dataset"
MODEL_REPO_ID = "BharatBoddu/tourism-purchase-model"

# Load train/test splits from Hugging Face
X_train = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/X_train.csv")
X_test = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/X_test.csv")
y_train = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/y_test.csv").values.ravel()

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 8, 12],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism_mlops_experiment")

with mlflow.start_run():
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1",
    )
    grid_search.fit(X_train, y_train)

    # Log every tuned parameter combination
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_f1", float(results["mean_test_score"][i]))

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    print("Best params:", grid_search.best_params_)
    print("Metrics:", metrics)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics(metrics)

    model_file = "best_tourism_model.joblib"
    joblib.dump(best_model, model_file)
    mlflow.log_artifact(model_file, artifact_path="model")

# Register model on Hugging Face Model Hub
api = HfApi(token=os.getenv("HF_TOKEN"))
try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
    print(f"Model repo '{MODEL_REPO_ID}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)
    print(f"Created model repo '{MODEL_REPO_ID}'.")

api.upload_file(
    path_or_fileobj="best_tourism_model.joblib",
    path_in_repo="best_tourism_model.joblib",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
)
print("Best model uploaded to Hugging Face Model Hub.")
