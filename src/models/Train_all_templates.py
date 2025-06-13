import os
import json
from pathlib import Path
from transformers import DonutProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset
import torch
torch.cuda.empty_cache()
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from train_donut import DonutPLModule  # make sure this is importable
from evaluate_donut import evaluate_donut  # reuse BLEU/ROUGE script

DATA_ROOT = Path("invoice_data")
IMAGE_DIR = Path("data/raw/images")
PRETRAINED_CKPT = "naver-clova-ix/donut-base"
MAX_EPOCHS = 5

for tid in range(1, 51):
    template_id = f"template_{tid:02d}"
    train_json_path = DATA_ROOT / template_id / "train.json"
    test_json_path = DATA_ROOT / template_id / "test.json"
    ckpt_dir = Path("checkpoints") / template_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Training on {template_id}...")

    # === Load Processor & Model ===
    processor = DonutProcessor.from_pretrained(PRETRAINED_CKPT)
    model = VisionEncoderDecoderModel.from_pretrained(PRETRAINED_CKPT)
    pl_model = DonutPLModule(processor, model)
    
    # === Prepare Datasets ===
    def load_dataset(json_path):
        with open(json_path) as f:
            samples = json.load(f)

        for s in samples:
            s["image_path"] = str(IMAGE_DIR / s["image"])

        return Dataset.from_list(samples)

    train_dataset = load_dataset(train_json_path)
    val_dataset = load_dataset(test_json_path)

    def collate_fn(batch):
        pixel_values, labels = [], []
        for sample in batch:
            image = processor.image_processor(image.open(sample["image_path"]).convert("RGB"), return_tensors="pt").pixel_values.squeeze(0)
            text = processor.tokenizer.tokenize(json.dumps(sample["ground_truth"], ensure_ascii=False))
            labels.append(processor.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).input_ids.squeeze(0))
            pixel_values.append(image)

        return {
            "pixel_values": torch.stack(pixel_values),
            "labels": torch.stack(labels),
        }

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

    

    # === Setup Trainer ===
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

    # === Evaluate
    ckpt_path = str(ckpt_dir / "best.ckpt")
    print(f"🧪 Evaluating {template_id}...")
    results = evaluate_donut(
        ckpt_path=ckpt_path,
        processor=processor,
        test_json=str(test_json_path),
        image_dir=str(IMAGE_DIR),
        output_path=str(ckpt_dir / "eval_metrics.json")
    )

    print(f"✅ {template_id} - Exact Match: {results['exact_match']:.2%}, BLEU: {results['bleu']:.4f}")
