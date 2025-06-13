# invoice_dataset.py

from torch.utils.data import Dataset
from PIL import Image
import json
import os

class InvoiceDataset(Dataset):
    def __init__(self, json_path, image_root, processor):
        with open(json_path, "r") as f:
            self.samples = json.load(f)
        self.image_root = image_root
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.image_root, item["image"])
        image = Image.open(image_path).convert("RGB")

        # task prompt and text target
        task_prompt = "invoice extraction"
        text_input = "<s_invoice>" + json.dumps(item["ground_truth"], ensure_ascii=False) + "</s_invoice>"

        encoding = self.processor(image, task_prompt, text_input, return_tensors="pt", padding="max_length", truncation=True, max_length = 512)
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding
