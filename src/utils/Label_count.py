import os
import json
from tqdm import tqdm
from collections import Counter

ANNOTATIONS_DIR = "data/raw/annotations"
label_counter = Counter()

for fname in tqdm(os.listdir(ANNOTATIONS_DIR)):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(ANNOTATIONS_DIR, fname)) as f:
        try:
            ann = json.load(f)
            for cls in ann.get("class_labels", []):
                label_counter[cls] += 1
        except Exception:
            continue

print("Unique labels and their frequencies:")
for label, count in sorted(label_counter.items()):
    print(f"{label}: {count}")
