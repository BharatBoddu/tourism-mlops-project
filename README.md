# Tourism MLOps Project

End-to-end MLOps pipeline for predicting whether a customer will purchase the **Wellness Tourism Package** — built by BharatBoddu for the AML & MLOps project.

## Project Structure

```
tourism_project/
├── data/                        # Raw tourism.csv (uploaded to HF dataset hub)
├── model_building/
│   ├── data_register.py         # Upload raw data to Hugging Face Dataset Hub
│   ├── prep.py                  # Clean, split and upload train/test data
│   └── train.py                 # Train model with GridSearch + MLflow tracking
├── deployment/
│   ├── app.py                   # Streamlit web application
│   ├── Dockerfile               # Container definition
│   └── requirements.txt         # App dependencies
├── hosting/
│   └── hosting.py               # Push deployment files to HF Space
└── requirements.txt             # Pipeline-level dependencies

.github/workflows/
└── pipeline.yml                 # GitHub Actions CI/CD workflow
```

## Rubric Coverage

| Criterion | Description |
|---|---|
| **Data Registration** | Creates HF dataset repo and uploads `tourism.csv` |
| **Data Preparation** | Loads from HF, cleans, splits (80/20), uploads splits back to HF |
| **Model Building + MLflow** | RandomForest with GridSearchCV, logs all params & metrics, registers best model on HF |
| **Model Deployment** | Dockerfile + Streamlit app + requirements.txt |
| **Hosting** | `hosting.py` pushes deployment folder to HF Space |
| **MLOps Pipeline** | GitHub Actions workflow with 4 sequential jobs |
| **Output Evaluation** | GitHub repo + HF Space links in notebook |

## Links

- **GitHub Repository**: https://github.com/BharatBoddu/tourism-mlops-project
- **Hugging Face Space**: https://huggingface.co/spaces/BharatBoddu/tourism-purchase-prediction-app
- **HF Dataset Hub**: https://huggingface.co/datasets/BharatBoddu/tourism-purchase-dataset
- **HF Model Hub**: https://huggingface.co/BharatBoddu/tourism-purchase-model

## How to Run

### Prerequisites
1. Add `HF_TOKEN` as a GitHub Secret (Settings → Secrets → Actions → New secret).
2. Ensure `tourism_project/data/tourism.csv` is present before triggering the pipeline.

### CI/CD Trigger
Push to the `main` branch → GitHub Actions automatically executes all 4 pipeline jobs in sequence:
1. `register-dataset` → Upload data to HF
2. `data-prep` → Clean & split data
3. `model-training` → Train, tune, log, register model
4. `deploy-hosting` → Push Streamlit app to HF Space
