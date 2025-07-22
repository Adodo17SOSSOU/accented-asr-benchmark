import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from utils.metrics import compute_metrics

PREDICTIONS_PATH = "models/outputs/whisper_predictions.json"
OUTPUT_PATH = "models/outputs/metrics_whisper.json"

def main():
    print("Loading predictions and references...")
    with open(PREDICTIONS_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples.")

    references = [sample["text"] for sample in data if sample.get("text") and sample.get("prediction")]
    hypotheses = [sample["prediction"] for sample in data if sample.get("text") and sample.get("prediction")]

    print("Computing metrics...")
    metrics = compute_metrics(references, hypotheses)

    print("Saving metrics...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Metrics saved to {OUTPUT_PATH}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
