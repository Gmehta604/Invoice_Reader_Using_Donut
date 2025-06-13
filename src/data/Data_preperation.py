import os
import json
from tqdm import tqdm

ANNOTATIONS_DIR = "data/raw/annotations"
OUTPUT_JSON_PATH = "data/processed/dount_all.json"

samples = []

for fname in tqdm(os.listdir(ANNOTATIONS_DIR)):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(ANNOTATIONS_DIR, fname)
    with open(path, "r") as f:
        try:
            ann = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Skipping malformed file: {fname}")
            continue

    image_id = os.path.splitext(fname)[0]  # e.g., Template1_Instance0
    fields = {}

    for key, value in ann.items():
        if isinstance(value, dict) and "text" in value:
            fields[key] = value["text"]

    if fields:
        samples.append({
            "image": f"{image_id}.jpg",
            "ground_truth": fields
        })

print(f"✅ Processed {len(samples)} valid annotation files")

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(samples, f, indent=2)
