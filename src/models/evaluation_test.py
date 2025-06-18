import torch
import json
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# === CONFIG ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "data/raw/images/Template1_Instance97.jpg"  # Replace with your image
MODEL_DIR = "checkpoints/final_hf"  # Path to saved fine-tuned model

# === Load model and processor ===
processor = DonutProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE).eval()

# === Load and preprocess image ===
image = Image.open(IMAGE_PATH).convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

# === Prepare decoder input (Donut-specific prompt) ===
task_prompt = "<s_invoice>"
decoder_input_ids = processor.tokenizer(
    task_prompt,
    add_special_tokens=False,
    return_tensors="pt"
).input_ids.to(DEVICE)

# === Generate output ===
output_ids = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=512
)
decoded_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

# === Try parsing as JSON ===
def try_parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": "Could not decode", "details": str(e), "raw_output": text}

# === Print prediction ===
parsed = try_parse_json(decoded_text)
print(json.dumps(parsed, indent=2))
