import os
import json
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    # Create the 'data/raw' directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)

    snapshot_download(
        repo_id="ariya2357/CORAL",
        repo_type="dataset",
        local_dir="data/raw"
    )