import os
from huggingface_hub import snapshot_download

# Create the 'data/raw' directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

snapshot_download(
    repo_id="ariya2357/CORAL",
    repo_type="dataset",
    local_dir="data/raw"
)