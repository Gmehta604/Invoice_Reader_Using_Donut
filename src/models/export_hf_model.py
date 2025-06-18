import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from train_donut import DonutPLModule

# === Paths ===
ckpt_path = "checkpoints/template_01/best.ckpt"
hf_save_path = "checkpoints/final_hf"
pretrained_ckpt = "naver-clova-ix/donut-base"

# === Load processor and base model ===
processor = DonutProcessor.from_pretrained(pretrained_ckpt)
base_model = VisionEncoderDecoderModel.from_pretrained(pretrained_ckpt)

# === Now load Lightning module using the base model
pl_model = DonutPLModule.load_from_checkpoint(ckpt_path, model=base_model).to("cpu").eval()

# === Save Hugging Face-compatible model and processor
processor.save_pretrained(hf_save_path)
pl_model.model.save_pretrained(hf_save_path)

print(f"✅ Hugging Face model + processor saved to: {hf_save_path}")
