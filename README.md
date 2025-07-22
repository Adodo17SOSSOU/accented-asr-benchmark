# Accented ASR Benchmark

This project benchmarks state-of-the-art ASR models on **French-accented English**, using Mozilla Common Voice data. We compare [Whisper](https://github.com/openai/whisper) and [Wav2Vec2](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) on Word and Character Error Rates.

---

## 🔍 Pipeline

1. Download & filter audio data  
2. Transcribe with Whisper / Wav2Vec2  
3. Evaluate with WER & CER  
4. Compare via leaderboard and plots

---

## 📁 Project Structure

- `data/`  
  └── `common_voice/en/` — filtered dataset  

- `models/`  
  └── `outputs/` — transcriptions and metrics  

- `scripts/`  
  ├── `download_data.py` — fetch and filter Common Voice  
  ├── `run_whisper.py` — run Whisper transcription  
  ├── `run_wav2vec.py` — run Wav2Vec2 transcription  
  ├── `compute_metrics_asr.py` — calculate WER/CER  
  └── `generate_leaderboard.py` — summarize results  

- `utils/`  
  ├── `metrics.py` — WER and CER helpers  
  ├── `audio.py` — optional audio preprocessing  
  └── `io.py` — JSON helpers  

- `results/`  
  ├── `leaderboard.csv` — WER/CER summary  
  ├── `wer_comparison.png` — WER plot  
  └── `cer_comparison.png` — CER plot  

- `requirements.txt` — dependencies  
- `run.sh` — optional full pipeline runner

---

## 📊 Results

| Model     | WER   | CER   | Samples |
|-----------|-------|-------|---------|
| Whisper   | 0.218 | 0.079 | 31      |
| Wav2Vec2  | 0.328 | 0.091 | 31      |

See `results/wer_comparison.png` and `results/cer_comparison.png` for visual comparison.

---

## 🚀 Run the Project

```bash
# 1. Setup environment
conda create -n asr_env python=3.10
conda activate asr_env
pip install -r requirements.txt

# 2. Download and prepare data
python scripts/download_data.py

# 3. Run inference
python scripts/run_whisper.py
python scripts/run_wav2vec.py

# 4. Compute metrics
python scripts/compute_metrics_asr.py

# 5. Generate leaderboard
python scripts/generate_leaderboard.py
