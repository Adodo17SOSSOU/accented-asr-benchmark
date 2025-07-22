import os
import json
import whisper
from tqdm import tqdm

INPUT_PATH = "data/common_voice/en/metadata.json"
OUTPUT_PATH = "models/outputs/whisper_predictions.json"
MODEL_SIZE = "medium"  # can be large or small

def main():
    print("Loading metadata...")
    with open(INPUT_PATH, "r") as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} samples.")
    print(f"Loading Whisper model ({MODEL_SIZE}) on cpu...")

    model = whisper.load_model("small", device="cpu")

    predictions = []

    for sample in tqdm(metadata, desc="Transcribing"):
        audio_path = sample["path"]
        try:
            result = model.transcribe(audio_path, language="en")
            predictions.append({
                "path": audio_path,
                "text": sample["text"],
                "prediction": result["text"]
            })
        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

