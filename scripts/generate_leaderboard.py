import os
import json
import pandas as pd

INPUT_DIR = "models/outputs"
OUTPUT_PATH = "results/leaderboard.csv"

def load_metrics(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    rows = []
    for filename in os.listdir(INPUT_DIR):
        if filename.startswith("metrics_") and filename.endswith(".json"):
            path = os.path.join(INPUT_DIR, filename)
            metrics = load_metrics(path)

            model_name = filename.replace("metrics_", "").replace(".json", "")
            rows.append({
                "model": model_name,
                "WER": round(metrics.get("WER", 0), 4),
                "CER": round(metrics.get("CER", 0), 4),
                "num_samples": metrics.get("num_samples", 0)
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("WER")
    df.to_csv(OUTPUT_PATH, index=False)

    print(f" Leaderboard saved to {OUTPUT_PATH}")
    print(df)

if __name__ == "__main__":
    main()
