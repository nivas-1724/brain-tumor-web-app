"""
Robustness Stress Test Experiment Script
Runs 7 controlled image degradation experiments on the test set and exports robustness_results.csv.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.preprocessing.image_preprocessing import load_split_into_memory
from backend.robustness.robustness_engine import evaluate_model_robustness

DATA_DIR = "dataset"
SAVED_MODELS_DIR = "backend/models/saved_models"
RESULTS_DIR = "results"
CSV_DIR = "results/csv"
os.makedirs(CSV_DIR, exist_ok=True)


def run_robustness_experiment():
    print("=" * 65)
    print(" EXPERIMENTAL FRAMEWORK: ROBUSTNESS & PERTURBATION STRESS TESTING")
    print("=" * 65)

    test_path = os.path.join(DATA_DIR, "Testing")
    X_test, y_test, _, _ = load_split_into_memory(test_path)

    model_path = os.path.join(SAVED_MODELS_DIR, "efficientnetb0.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(SAVED_MODELS_DIR, "baseline_cnn.h5")

    print(f" [MODEL] Loading model for robustness evaluation: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    results = evaluate_model_robustness(model, X_test, y_test, num_samples=300)

    print("\n" + "─" * 60)
    print(" TRANSFORMATION | ACCURACY | F1 SCORE | CONF DENSITY | STABILITY")
    print("─" * 60)
    for r in results:
        print(f" {r['transformation']:<14} | {r['accuracy']:>6.2f}%  | {r['f1_score']:>6.2f}%  | {r['avg_confidence']:>6.2f}%       | {r['stability_pct']:>6.2f}%")
    print("─" * 60)

    df = pd.DataFrame(results)
    csv_path = os.path.join(CSV_DIR, "robustness_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n [SAVED] robustness_results.csv -> {csv_path}")

    with open(os.path.join(RESULTS_DIR, "robustness_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_robustness_experiment()
