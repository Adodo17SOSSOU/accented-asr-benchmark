import pandas as pd
import os
import json

SOURCE_TSV = "data/common_voice/raw/common_voice_16_en/en/validated.tsv"
AUDIO_BASE_PATH = "data/common_voice/raw/common_voice_16_en/en/clips"
OUTPUT_PATH = "data/common_voice/en/metadata.json"
MAX_SAMPLES = 200

def main():
    print("Loading validated.tsv...")
    df = pd.read_csv(SOURCE_TSV, sep="\t")

    print("Filtering French-accented English samples...")
    df_filtered = df[
        df["accents"].fillna("").str.contains("french", case=False)
        & df["path"].notna()
        & df["sentence"].notna()
    ]

    print(f"Found {len(df_filtered)} matching samples.")

    metadata = [
        {"path": os.path.join(AUDIO_BASE_PATH, row["path"]), "text": row["sentence"]}
        for _, row in df_filtered.head(MAX_SAMPLES).iterrows()
    ]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(metadata)} samples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
