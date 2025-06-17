import os
import json
from pathlib import Path
from PIL import Image
from functools import partial

import torch
torch.cuda.empty_cache()
from torch.utils.data import DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from train_donut import DonutPLModule
from evaluate_donut import evaluate_donut

# === CONFIG ===
DATA_ROOT = Path("invoice_data")
IMAGE_DIR = Path("data/raw/images")
PRETRAINED_CKPT = "naver-clova-ix/donut-base"
MAX_EPOCHS = 5
BATCH_SIZE = 2
NUM_WORKERS = 8


def collate_fn(batch, processor):
    pixel_values, labels = [], []
    for sample in batch:
        image = processor.image_processor(
            Image.open(sample["image_path"]).convert("RGB"),
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        text = json.dumps(sample["ground_truth"], ensure_ascii=False)
        input_ids = processor.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512  # 👈 Set a max_length here
        ).input_ids.squeeze(0)

        pixel_values.append(image)
        labels.append(input_ids)

    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": torch.stack(labels),
    }


def main():
    trained_templates = []

    # === PHASE 1: TRAIN ALL TEMPLATES ===
    for tid in range(1, 51):
        template_id = f"template_{tid:02d}"
        train_json_path = DATA_ROOT / template_id / "train.json"
        test_json_path = DATA_ROOT / template_id / "test.json"
        ckpt_dir = Path("checkpoints") / template_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if not train_json_path.exists() or not test_json_path.exists():
            print(f"⚠️ Skipping {template_id} due to missing train/test files")
            continue

        print(f"\n📦 Training on {template_id}...")

        # Load Processor & Model
        processor = DonutProcessor.from_pretrained(PRETRAINED_CKPT)
        model = VisionEncoderDecoderModel.from_pretrained(PRETRAINED_CKPT)
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        pl_model = DonutPLModule(model)

        # Load Datasets
        def load_dataset(json_path):
            with open(json_path) as f:
                samples = json.load(f)
            for s in samples:
                s["image_path"] = str(IMAGE_DIR / s["image"])
            return samples

        train_samples = load_dataset(train_json_path)
        val_samples = load_dataset(test_json_path)

        train_loader = DataLoader(
            train_samples,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=partial(collate_fn, processor=processor),
            num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_samples,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=partial(collate_fn, processor=processor),
            num_workers=NUM_WORKERS
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )

        trainer = Trainer(
            max_epochs=MAX_EPOCHS,
            precision="16-mixed",
            default_root_dir=ckpt_dir,
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            devices=1
        )

        trainer.fit(pl_model, train_loader, val_loader)
        trained_templates.append(template_id)

    # === PHASE 2: EVALUATE ALL TRAINED TEMPLATES ===
    for template_id in trained_templates:
        print(f"🧪 Evaluating {template_id}...")
        processor = DonutProcessor.from_pretrained(PRETRAINED_CKPT)

        test_json_path = DATA_ROOT / template_id / "test.json"
        ckpt_path = (Path("checkpoints") / template_id / "best.ckpt").as_posix()
        output_path = Path("checkpoints") / template_id / "eval_metrics.json"

        results = evaluate_donut(
            ckpt_path=ckpt_path,
            processor=processor,
            test_json=str(test_json_path),
            image_dir=str(IMAGE_DIR),
            output_path=str(output_path)
        )

        print(f"✅ {template_id} - Exact Match: {results['exact_match']:.2%}, BLEU: {results['bleu']:.4f}")



if __name__ == "__main__":
    main()
