from huggingface_hub import HfApi
import os

SPACE_REPO_ID = "BharatBoddu/tourism-purchase-prediction-app"

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    path_in_repo="",
)
print("Deployment files pushed to Hugging Face Space.")
