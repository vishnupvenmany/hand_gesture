"""
Dataset Downloader for SignConnect Live (Phase 1)
-------------------------------------------------
Downloads the 32-class ISL hand landmark dataset (train.csv, val.csv, test.csv, class_mapping.json)
from the official repository into phase1/data/raw/.
"""

import os
import urllib.request

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

FILES = {
    "train.csv": "https://raw.githubusercontent.com/Jeel-Pipaliya/SignBridge-AI/main/data/processed/train.csv",
    "val.csv": "https://raw.githubusercontent.com/Jeel-Pipaliya/SignBridge-AI/main/data/processed/val.csv",
    "test.csv": "https://raw.githubusercontent.com/Jeel-Pipaliya/SignBridge-AI/main/data/processed/test.csv",
    "class_mapping.json": "https://raw.githubusercontent.com/Jeel-Pipaliya/SignBridge-AI/main/data/class_mapping.json"
}

def download_dataset():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"Downloading ISL dataset files to: {RAW_DATA_DIR}")

    headers = {'User-Agent': 'Mozilla/5.0'}

    for filename, url in FILES.items():
        destination_path = os.path.join(RAW_DATA_DIR, filename)
        print(f"Fetching {filename}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(destination_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"  [SUCCESS] Saved {filename} ({os.path.getsize(destination_path)} bytes)")
        except Exception as e:
            print(f"  [ERROR] Failed to download {filename}: {e}")
            raise e

    print("Dataset download complete!")

if __name__ == "__main__":
    download_dataset()
