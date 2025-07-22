# scripts/run_wav2vec.py

import os
import json
from tqdm import tqdm
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

INPUT_PATH = "data/common_voice/en/metadata.json"
OUTPUT_PATH = "models/outputs/wav2vec_predictions.json"
MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform.squeeze(), sample_rate

def main():
    print("Loading metadata...")
    with open(INPUT_PATH, "r") as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} samples.")
    print(f"Loading Wav2Vec2 model on {DEVICE}...")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    predictions = []

    for sample in tqdm(metadata, desc="Transcribing"):
        try:
            waveform, sample_rate = load_audio(sample["path"])
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
            predictions.append({
                "path": sample["path"],
                "text": sample["text"],
                "prediction": transcription
            })
        except Exception as e:
            print(f"Failed to process {sample['path']}: {e}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
