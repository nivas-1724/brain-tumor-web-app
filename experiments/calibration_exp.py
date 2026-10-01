"""
Calibration Experiment Script
Runs Temperature Scaling on validation logits, evaluates ECE before and after calibration, and exports calibration_results.csv.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.preprocessing.image_preprocessing import load_split_into_memory
from backend.calibration.temperature_scaling import TemperatureScaler, compute_ece

DATA_DIR = "dataset"
SAVED_MODELS_DIR = "backend/models/saved_models"
RESULTS_DIR = "results"
CSV_DIR = "results/csv"
os.makedirs(CSV_DIR, exist_ok=True)


def run_calibration_experiment():
    print("=" * 65)
    print(" EXPERIMENTAL FRAMEWORK: PROBABILITY CALIBRATION & ECE EVALUATION")
    print("=" * 65)

    # 1. Load validation split
    train_path = os.path.join(DATA_DIR, "Training")
    X_train_full, y_train_full, _, _ = load_split_into_memory(train_path)
    _, X_val, _, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=np.argmax(y_train_full, axis=1)
    )

    # 2. Load independent test set
    test_path = os.path.join(DATA_DIR, "Testing")
    X_test, y_test, _, _ = load_split_into_memory(test_path)
    y_test_indices = np.argmax(y_test, axis=1)

    # Select best model (EfficientNetB0 or available)
    model_path = os.path.join(SAVED_MODELS_DIR, "efficientnetb0.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(SAVED_MODELS_DIR, "baseline_cnn.h5")

    print(f" [MODEL] Loading model for calibration: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    # Get validation probabilities & pseudo-logits
    val_probs = model.predict(X_val, verbose=0)
    eps = 1e-12
    val_logits = np.log(np.clip(val_probs, eps, 1.0 - eps))
    y_val_indices = np.argmax(y_val, axis=1)

    # Fit Temperature Scaler
    scaler = TemperatureScaler()
    opt_temp = scaler.fit(val_logits, y_val_indices)
    print(f"   Optimal Temperature Parameter (T) : {opt_temp:.4f}")

    # Evaluate on test set before and after calibration
    test_raw_probs = model.predict(X_test, verbose=0)
    test_cal_probs = scaler.calibrate(test_raw_probs)

    ece_before, bin_data_before = compute_ece(test_raw_probs, y_test_indices, n_bins=10)
    ece_after, bin_data_after = compute_ece(test_cal_probs, y_test_indices, n_bins=10)

    print(f"   ECE Before Calibration : {ece_before * 100.0:.2f}%")
    print(f"   ECE After Calibration  : {ece_after * 100.0:.2f}%")

    calibration_summary = {
        "temperature_T": round(opt_temp, 4),
        "ece_before_pct": round(ece_before * 100.0, 2),
        "ece_after_pct": round(ece_after * 100.0, 2),
        "bins_before": bin_data_before,
        "bins_after": bin_data_after,
    }

    # Export calibration_results.csv
    csv_rows = [
        {"Stage": "Before Calibration", "ECE_pct": round(ece_before * 100.0, 2), "Temperature_T": 1.0},
        {"Stage": "After Calibration", "ECE_pct": round(ece_after * 100.0, 2), "Temperature_T": round(opt_temp, 4)},
    ]
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(CSV_DIR, "calibration_results.csv")
    df.to_csv(csv_path, index=False)
    print(f" [SAVED] calibration_results.csv -> {csv_path}")

    with open(os.path.join(RESULTS_DIR, "calibration_summary.json"), "w") as f:
        json.dump(calibration_summary, f, indent=2)

    return calibration_summary


if __name__ == "__main__":
    run_calibration_experiment()
