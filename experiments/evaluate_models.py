"""
Model Evaluation Experiment Script
Evaluates Baseline Custom CNN, ResNet50, EfficientNetB0, MobileNetV2, and Ensemble Model on real independent test set.
Generates model_comparison.csv and saves confusion matrix plots.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.preprocessing.image_preprocessing import load_split_into_memory, CLASSES
from backend.evaluation.metrics import evaluate_predictions
from backend.ensemble.ensemble_engine import EnsemblePredictor

DATA_DIR = "dataset"
SAVED_MODELS_DIR = "backend/models/saved_models"
RESULTS_DIR = "results"
CSV_DIR = "results/csv"
CM_DIR = "results/confusion_matrices"
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)


def plot_and_save_cm(cm, model_name, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_evaluation_experiment():
    print("=" * 65)
    print(" EXPERIMENTAL FRAMEWORK: EVALUATING MODELS ON TEST SET (1,600 IMAGES)")
    print("=" * 65)

    test_path = os.path.join(DATA_DIR, "Testing")
    X_test, y_test, filenames, filepaths = load_split_into_memory(test_path)
    print(f" [DATA] Loaded {len(X_test)} independent test set images.")

    models_info = {
        "Baseline_CNN": "baseline_cnn.h5",
        "ResNet50": "resnet50.h5",
        "EfficientNetB0": "efficientnetb0.h5",
        "MobileNetV2": "mobilenetv2.h5",
    }

    loaded_models = {}
    evaluation_results = {}
    f1_weights = {}

    for name, filename in models_info.items():
        fpath = os.path.join(SAVED_MODELS_DIR, filename)
        if not os.path.exists(fpath):
            print(f" ⚠️ Model file missing: {fpath}. Skipping...")
            continue

        print(f"\n [EVAL] Evaluating {name}...")
        model = keras.models.load_model(fpath, compile=False)
        loaded_models[name] = model

        # Benchmark inference time
        t0 = time.time()
        preds = model.predict(X_test, batch_size=32, verbose=0)
        total_time = time.time() - t0
        avg_inf_time_ms = (total_time / len(X_test)) * 1000.0

        metrics = evaluate_predictions(y_test, preds)
        metrics["inference_time_ms"] = round(avg_inf_time_ms, 2)
        metrics["parameters"] = int(model.count_params())

        evaluation_results[name] = metrics
        f1_weights[name] = metrics["f1_weighted"]

        # Plot confusion matrix
        plot_path = os.path.join(CM_DIR, f"cm_{name.lower()}.png")
        plot_and_save_cm(np.array(metrics["confusion_matrix"]), name, plot_path)
        print(f"   Acc: {metrics['accuracy']}%, F1: {metrics['f1_weighted']}%, Inf: {avg_inf_time_ms:.2f}ms/img, Params: {metrics['parameters']:,}")

    # ─────────────────────────────────────────────
    # Evaluate Ensemble Model
    # ─────────────────────────────────────────────
    if len(loaded_models) >= 2:
        print("\n [EVAL] Evaluating Ensemble Model (Weighted Probability Averaging)...")
        ensemble = EnsemblePredictor(models_dict=loaded_models, weights_dict=f1_weights)

        t0 = time.time()
        ens_preds = ensemble.predict_probs(X_test, method="weighted")
        total_time = time.time() - t0
        avg_inf_time_ms = (total_time / len(X_test)) * 1000.0

        ens_metrics = evaluate_predictions(y_test, ens_preds)
        ens_metrics["inference_time_ms"] = round(avg_inf_time_ms, 2)
        ens_metrics["parameters"] = sum(m.count_params() for m in loaded_models.values())

        evaluation_results["Ensemble"] = ens_metrics

        plot_path = os.path.join(CM_DIR, "cm_ensemble.png")
        plot_and_save_cm(np.array(ens_metrics["confusion_matrix"]), "Ensemble Model", plot_path)
        print(f"   Ensemble Acc: {ens_metrics['accuracy']}%, F1: {ens_metrics['f1_weighted']}%, Inf: {avg_inf_time_ms:.2f}ms/img")

    # ─────────────────────────────────────────────
    # Identify Best Single Model
    # ─────────────────────────────────────────────
    single_models = {k: v for k, v in evaluation_results.items() if k != "Ensemble"}
    best_model_name = max(single_models.items(), key=lambda item: item[1]["f1_weighted"])[0]
    best_model = single_models[best_model_name]

    print("\n" + "=" * 65)
    print(f" 🏆 AUTOMATICALLY SELECTED BEST SINGLE MODEL: {best_model_name}")
    print(f"    Validation/Test F1 Score : {best_model['f1_weighted']}%")
    print(f"    Glioma Recall            : {best_model['per_class']['glioma']['recall']}%")
    print(f"    Meningioma Recall        : {best_model['per_class']['meningioma']['recall']}%")
    print(f"    No Tumor Recall          : {best_model['per_class']['notumor']['recall']}%")
    print(f"    Pituitary Recall         : {best_model['per_class']['pituitary']['recall']}%")
    print("=" * 65)

    # ─────────────────────────────────────────────
    # Save model_comparison.csv
    # ─────────────────────────────────────────────
    csv_rows = []
    for model_name, res in evaluation_results.items():
        csv_rows.append({
            "Model": model_name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision_weighted"],
            "Recall": res["recall_weighted"],
            "F1_Score": res["f1_weighted"],
            "Glioma_Recall": res["per_class"]["glioma"]["recall"],
            "Meningioma_Recall": res["per_class"]["meningioma"]["recall"],
            "NoTumor_Recall": res["per_class"]["notumor"]["recall"],
            "Pituitary_Recall": res["per_class"]["pituitary"]["recall"],
            "Parameters": res["parameters"],
            "Inference_Time_ms": res["inference_time_ms"],
        })

    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(CSV_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n [SAVED] model_comparison.csv -> {csv_path}")

    with open(os.path.join(RESULTS_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(evaluation_results, f, indent=2)

    return evaluation_results, best_model_name


if __name__ == "__main__":
    run_evaluation_experiment()
