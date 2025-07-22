from jiwer import wer, cer
import numpy as np
import editdistance

def compute_metrics(references, hypotheses):
    return {
        "WER": wer(references, hypotheses),
        "CER": cer(references, hypotheses),
        "num_samples": len(references)
    }

def compute_wer(predictions, references):
    """
    Compute Word Error Rate (WER)
    """
    total_words = 0
    total_errors = 0

    for pred, ref in zip(predictions, references):
        pred_words = pred.strip().lower().split()
        ref_words = ref.strip().lower().split()
        total_words += len(ref_words)
        total_errors += editdistance.eval(pred_words, ref_words)

    if total_words == 0:
        return 0.0

    return total_errors / total_words

def compute_cer(predictions, references):
    """
    Compute Character Error Rate (CER)
    """
    total_chars = 0
    total_errors = 0

    for pred, ref in zip(predictions, references):
        pred_chars = list(pred.strip().lower())
        ref_chars = list(ref.strip().lower())
        total_chars += len(ref_chars)
        total_errors += editdistance.eval(pred_chars, ref_chars)

    if total_chars == 0:
        return 0.0

    return total_errors / total_chars
