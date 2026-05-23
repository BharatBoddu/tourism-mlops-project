from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

DATASET_REPO_ID = "BharatBoddu/tourism-purchase-dataset"
DATASET_REPO_TYPE = "dataset"
LOCAL_DATA_PATH = "tourism_project/data"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=DATASET_REPO_ID, repo_type=DATASET_REPO_TYPE)
    print(f"Dataset repo '{DATASET_REPO_ID}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=DATASET_REPO_ID, repo_type=DATASET_REPO_TYPE, private=False)
    print(f"Created dataset repo '{DATASET_REPO_ID}'.")

api.upload_folder(
    folder_path=LOCAL_DATA_PATH,
    repo_id=DATASET_REPO_ID,
    repo_type=DATASET_REPO_TYPE,
)
print("Data uploaded to Hugging Face Dataset Hub.")
