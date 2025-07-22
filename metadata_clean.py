import os
import json

metadata_path = "data/common_voice/en/metadata.json"
audio_dir = "data/common_voice/raw/common_voice_16_en/en/clips/"

# Load the current metadata
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Keep only entries with actual files present
cleaned_metadata = [
    sample for sample in metadata
    if os.path.exists(sample["path"])
]

print(f"Before: {len(metadata)} entries")
print(f"After: {len(cleaned_metadata)} entries")

# Save cleaned metadata
with open(metadata_path, "w") as f:
    json.dump(cleaned_metadata, f, indent=2)

print("Cleaned metadata saved.")
