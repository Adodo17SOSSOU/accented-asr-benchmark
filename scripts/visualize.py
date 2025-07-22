import pandas as pd
import matplotlib.pyplot as plt
import os

# Data for leaderboard
data = {
    "Model": ["Whisper", "Wav2Vec2"],
    "WER": [0.2185, 0.3278],
    "CER": [0.0785, 0.0913]
}

df = pd.DataFrame(data)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# File paths
wer_plot_path = "results/wer_comparison.png"
cer_plot_path = "results/cer_comparison.png"

# Plot WER
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["WER"], color="skyblue")
plt.title("Word Error Rate (WER) Comparison")
plt.ylabel("WER")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig(wer_plot_path)
plt.close()

# Plot CER
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["CER"], color="salmon")
plt.title("Character Error Rate (CER) Comparison")
plt.ylabel("CER")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig(cer_plot_path)
plt.close()

print(f"Saved WER plot to {wer_plot_path}")
print(f"Saved CER plot to {cer_plot_path}")
