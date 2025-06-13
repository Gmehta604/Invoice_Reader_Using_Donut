import os
import json
import random
from pathlib import Path

annotation_dir = Path("data/raw/annotations")
image_dir = Path("data/raw/images")
output_dir = Path("invoice_data")

output_dir.mkdir(parents=True, exist_ok=True)

template_count = 50
instances_per_template = 200
train_ratio = 0.8

for template_id in range(1, template_count + 1):
    template_name = f"Template{template_id}"
    json_files = sorted(annotation_dir.glob(f"{template_name}_Instance*.json"))
    
    assert len(json_files) == instances_per_template, f"Expected 200 files for {template_name}, got {len(json_files)}"

    random.shuffle(json_files)
    split_idx = int(train_ratio * instances_per_template)

    train_files = json_files[:split_idx]
    test_files = json_files[split_idx:]

    def convert(files):
        data = []
        for file in files:
            with open(file, "r") as f:
                ann = json.load(f)
            image_name = file.with_suffix(".jpg").name
            fields = {k: v["text"] for k, v in ann.items() if k != "TABLE" and k != "OTHER" and isinstance(v, dict) and "text" in v}
            data.append({
                "image": image_name,
                "ground_truth": fields
            })
        return data

    template_folder = output_dir / f"template_{template_id:02d}"
    template_folder.mkdir(parents=True, exist_ok=True)

    with open(template_folder / "train.json", "w") as f:
        json.dump(convert(train_files), f, indent=2)

    with open(template_folder / "test.json", "w") as f:
        json.dump(convert(test_files), f, indent=2)

    print(f"✅ Split Template {template_id}: train={len(train_files)}, test={len(test_files)}")
