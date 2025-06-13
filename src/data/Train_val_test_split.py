import json
import random
import os

# === CONFIG ===
INPUT_JSON = "data/processed/dount_all.json"         # Path to your full dataset
OUTPUT_DIR = "data/processed"                        # Where train/val/test will be saved
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# === LOAD DATA ===
with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

# === SHUFFLE AND SPLIT ===
random.seed(RANDOM_SEED)
random.shuffle(data)

n = len(data)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

# === SAVE SPLITS ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "train.json"), 'w') as f:
    json.dump(train_data, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "val.json"), 'w') as f:
    json.dump(val_data, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "test.json"), 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"✅ Done! Saved {len(train_data)} train, {len(val_data)} val, and {len(test_data)} test samples.")
