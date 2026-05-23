# Tourism MLOps Project — Setup & Continuation Guide

**GitHub Repo:** https://github.com/BharatBoddu/tourism-mlops-project  
**HF Space:** https://huggingface.co/spaces/BharatBoddu/tourism-purchase-prediction-app  

---

## Step 1 — Prerequisites (install on the new system)

```bash
# Python 3.9+ required
python --version

# Install Git
git --version

# Install required Python packages
pip install huggingface_hub pandas scikit-learn joblib mlflow streamlit
```

---

## Step 2 — Clone the repo

```bash
git clone https://github.com/BharatBoddu/tourism-mlops-project.git
cd tourism-mlops-project
```

---

## Step 3 — Set up Hugging Face login

Go to https://huggingface.co/settings/tokens → create a **Write** token, then run:

```bash
huggingface-cli login
# paste your HF token when prompted
```

Or set it as an environment variable:

```bash
# Windows CMD
set HF_TOKEN=your_hf_token_here

# Windows PowerShell
$env:HF_TOKEN = "your_hf_token_here"

# Linux/Mac
export HF_TOKEN=your_hf_token_here
```

---

## Step 4 — Run the pipeline manually (step by step)

Run each script in order from the repo root:

```bash
# 1. Register dataset on HF
python tourism_project/model_building/data_register.py

# 2. Prepare data (clean + split + upload to HF)
python tourism_project/model_building/prep.py

# 3. Start MLflow tracking server (open a separate terminal)
mlflow ui --port 5000
# Open http://localhost:5000 in your browser to view experiments

# 4. Train model (run in the original terminal)
python tourism_project/model_building/train.py

# 5. Push Streamlit app to HF Space
python tourism_project/hosting/hosting.py
```

---

## Step 5 — Run the Streamlit app locally (optional test)

```bash
cd tourism_project/deployment
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501 in your browser
```

---

## Step 6 — Run the notebook for submission

Open `Learner_Template_Notebook_AML_and_MLOps_Project_Completed.ipynb` in **Google Colab**:

1. Go to https://colab.research.google.com → **File → Upload notebook**
2. Upload the `Learner_Template_Notebook_AML_and_MLOps_Project_Completed.ipynb` file
3. Upload `tourism.csv` using the Files panel on the left
4. Add this cell at the top of the notebook and run it first:
   ```python
   import os
   os.environ['HF_TOKEN'] = 'your_hf_token_here'
   ```
5. Fill in the GitHub token where the placeholder `<--------GitHub Token--------->` appears
6. **Runtime → Run all** — execute all cells top to bottom
7. Download the executed notebook (File → Download → .ipynb) for submission

---

## Step 7 — Monitor GitHub Actions pipeline

Go to: https://github.com/BharatBoddu/tourism-mlops-project/actions

The pipeline runs automatically on every push to `main`. It has 4 jobs:

| Job | Script | What it does |
|---|---|---|
| register-dataset | data_register.py | Uploads tourism.csv to HF Dataset Hub |
| data-prep | prep.py | Cleans data, creates train/test splits on HF |
| model-training | train.py | Trains RandomForest, logs to MLflow, registers model on HF |
| deploy-hosting | hosting.py | Pushes Streamlit app to HF Space |

If a job fails, click on it in the Actions tab to view logs.

---

## Key URLs

| Resource | URL |
|---|---|
| GitHub Repo | https://github.com/BharatBoddu/tourism-mlops-project |
| HF Space (Streamlit app) | https://huggingface.co/spaces/BharatBoddu/tourism-purchase-prediction-app |
| HF Dataset | https://huggingface.co/datasets/BharatBoddu/tourism-purchase-dataset |
| HF Model | https://huggingface.co/BharatBoddu/tourism-purchase-model |
| GitHub Actions | https://github.com/BharatBoddu/tourism-mlops-project/actions |
| MLflow UI (local) | http://localhost:5000 |
| Streamlit (local) | http://localhost:8501 |

---

## Project Folder Structure

```
tourism-mlops-project/
├── .github/
│   └── workflows/
│       └── pipeline.yml          # GitHub Actions CI/CD (4 jobs)
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv           # Raw dataset
│   ├── model_building/
│   │   ├── data_register.py      # Register data on HF
│   │   ├── prep.py               # Data prep + upload splits
│   │   └── train.py              # Train + MLflow + register model
│   ├── deployment/
│   │   ├── app.py                # Streamlit prediction app
│   │   ├── requirements.txt      # App dependencies
│   │   └── Dockerfile            # Container for HF Space
│   ├── hosting/
│   │   └── hosting.py            # Push app to HF Space
│   └── requirements.txt          # Pipeline dependencies
├── Learner_Template_Notebook_AML_and_MLOps_Project_Completed.ipynb
├── tourism.csv
└── README.md
```

---

## Rubric Checklist

| Criterion | Points | Status |
|---|---|---|
| Data Registration (folder + HF upload) | 3 | ✅ data_register.py |
| Data Preparation (load HF, clean, split, upload) | 7 | ✅ prep.py |
| Model Building + MLflow tracking | 13 | ✅ train.py |
| Model Deployment (Dockerfile, Streamlit, deps, hosting script) | 11 | ✅ app.py + Dockerfile + hosting.py |
| MLOps Pipeline (GitHub Actions YAML, push, automate) | 15 | ✅ pipeline.yml |
| Output Evaluation (GitHub + HF Space links/screenshots) | 4 | ⬜ Add screenshots to notebook |
| Notebook Quality (structure, comments, no errors, executed) | 7 | ⬜ Run notebook in Colab |
| **Total** | **60** | |

> **Important:** Rubric states — *"If the code is correct but not executed, 50% of the marks will be deducted."*  
> Make sure to run the notebook in Colab and download the executed version for submission.
