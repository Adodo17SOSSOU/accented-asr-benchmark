import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from utils.metrics import compute_wer, compute_cer

PREDICTIONS_PATH = "models/outputs/wav2vec_predictions.json"
OUTPUT_PATH = "models/outputs/metrics_wav2vec2.json"

def main():
    print("Loading predictions and references...")

    with open(PREDICTIONS_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples.")

    references = [entry["text"] for entry in data]
    predictions = [entry["prediction"] for entry in data]

    print("Computing metrics...")
    wer = compute_wer(predictions, references)
    cer = compute_cer(predictions, references)

    metrics = {
        "WER": wer,
        "CER": cer,
        "num_samples": len(data)
    }

    print("Saving metrics...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Metrics saved to {OUTPUT_PATH}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

