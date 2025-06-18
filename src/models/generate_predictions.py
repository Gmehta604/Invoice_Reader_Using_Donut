import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel

from train_donut import DonutPLModule

# === CONFIG ===
DATA_ROOT = Path("invoice_data")
IMAGE_DIR = Path("data/raw/images")
CKPT_DIR = Path("checkpoints")
PRETRAINED_CKPT = "naver-clova-ix/donut-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1


def generate_predictions(template_id: str):
    print(f"🔍 Generating predictions for {template_id}...")

    test_json = DATA_ROOT / template_id / "test.json"
    ckpt_path = CKPT_DIR / template_id / "best.ckpt"
    output_path = CKPT_DIR / template_id / "predictions.json"

    # === Load model & processor ===
    processor = DonutProcessor.from_pretrained(PRETRAINED_CKPT)
    base_model = VisionEncoderDecoderModel.from_pretrained(PRETRAINED_CKPT)
    pl_model = DonutPLModule.load_from_checkpoint(str(ckpt_path), model=base_model)
    pl_model = pl_model.to(DEVICE).eval()

    # === Load test samples ===
    with open(test_json) as f:
        test_samples = json.load(f)

    predictions = []

    for sample in tqdm(test_samples):
        image_path = IMAGE_DIR / sample["image"]
        image = Image.open(image_path).convert("RGB")

        # Preprocess image
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

        # Generate output
        try:
            generated_ids = pl_model.model.generate(pixel_values, max_length=512)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            predictions.append({
                "image": sample["image"],
                "ground_truth": sample["ground_truth"],
                "prediction": json.loads(generated_text)
            })
        except Exception as e:
            predictions.append({
                "image": sample["image"],
                "ground_truth": sample["ground_truth"],
                "prediction": {"error": "Could not decode", "details": str(e)}
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved predictions to: {output_path}")


if __name__ == "__main__":
    for tid in range(1, 26):
        template_id = f"template_{tid:02d}"
        ckpt_file = CKPT_DIR / template_id / "best.ckpt"
        test_file = DATA_ROOT / template_id / "test.json"

        if not ckpt_file.exists() or not test_file.exists():
            print(f"⚠️ Skipping {template_id} due to missing checkpoint or test data.")
            continue

        generate_predictions(template_id)
