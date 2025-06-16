import json
from pathlib import Path
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
from evaluate import load
import torch
from tqdm import tqdm


def evaluate_donut(ckpt_path, processor, test_json, image_dir, output_path):
    # Load fine-tuned model
    model = VisionEncoderDecoderModel.from_pretrained(ckpt_path).to("cuda")
    model.eval()

    # Load test samples
    with open(test_json, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    bleu = load("bleu")
    rouge = load("rouge")

    exact_matches = 0
    predictions = []
    references = []

    for sample in tqdm(test_samples, desc="Evaluating"):
        image_path = str(Path(image_dir) / sample["image"])
        image = Image.open(image_path).convert("RGB")
        task_prompt = "invoice extraction"

        # Prepare input
        pixel_values = processor(image, task_prompt, return_tensors="pt").pixel_values.to("cuda")

        # Run model
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=512, num_beams=2)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Prepare reference
        ground_truth_text = json.dumps(sample["ground_truth"], ensure_ascii=False).strip()
        predictions.append(generated_text)
        references.append([ground_truth_text])

        if generated_text == ground_truth_text:
            exact_matches += 1

    # Evaluate
    exact_match_score = exact_matches / len(test_samples)
    bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
    rouge_score = rouge.compute(predictions=predictions, references=[ref[0] for ref in references])

    # Save results
    result = {
        "exact_match": exact_match_score,
        "bleu": bleu_score,
        "rouge": {
            "rouge1": rouge_score["rouge1"].mid.fmeasure,
            "rouge2": rouge_score["rouge2"].mid.fmeasure,
            "rougeL": rouge_score["rougeL"].mid.fmeasure,
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result
