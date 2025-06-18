# evaluate_structured_invoice.py

import json
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path

def normalize(text):
    return text.strip().lower() if isinstance(text, str) else ""

def field_level_evaluation(pred_data, ref_data, fields):
    total_samples = len(pred_data)
    exact_match_count = 0

    all_preds = {field: [] for field in fields}
    all_refs = {field: [] for field in fields}

    for pred, ref in zip(pred_data, ref_data):
        pred_fields = pred if isinstance(pred, dict) else pred.get("ground_truth", {})
        ref_fields = ref.get("ground_truth", {})

        matched_all = True
        for field in fields:
            p_val = normalize(pred_fields.get(field, ""))
            r_val = normalize(ref_fields.get(field, ""))
            all_preds[field].append(p_val)
            all_refs[field].append(r_val)
            if p_val != r_val:
                matched_all = False

        if matched_all:
            exact_match_count += 1

    print(f"\n🔎 Exact Match Accuracy: {exact_match_count}/{total_samples} = {exact_match_count / total_samples:.2%}")
    print("\n📊 Per-Field F1 Score:")

    for field in fields:
        preds = all_preds[field]
        refs = all_refs[field]
        p, r, f1, _ = precision_recall_fscore_support(refs, preds, average='micro', zero_division=0)
        print(f"  - {field}: F1 = {f1:.4f}, Precision = {p:.4f}, Recall = {r:.4f}")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    reference_path = Path("invoice_data/template_01/test.json")
    prediction_path = Path("checkpoints/template_01/predictions.json")

    references = load_json(reference_path)
    predictions = load_json(prediction_path)

    fields_to_eval = [
        "BUYER", "DATE", "DISCOUNT", "DUE_DATE", "GSTIN_BUYER", "NOTE", "PAYMENT_DETAILS",
        "PO_NUMBER", "SELLER_ADDRESS", "SELLER_EMAIL", "SELLER_SITE", "SUB_TOTAL", "TAX",
        "TITLE", "TOTAL", "TOTAL_WORDS", "GSTIN_SELLER"
    ]

    field_level_evaluation(predictions, references, fields_to_eval)
