"""
Error Analysis & Misclassification Investigation Script
Investigates Glioma vs Meningioma confusion on real test set and exports misclassification_results.csv & prediction_results.csv.
"""

import os
import sys
import json
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.preprocessing.image_preprocessing import load_split_into_memory, CLASSES
from backend.evaluation.metrics import evaluate_predictions
from backend.explainability.multi_xai import generate_gradcam_heatmap, heatmap_to_overlay, pil_to_base64_uri

DATA_DIR = "dataset"
SAVED_MODELS_DIR = "backend/models/saved_models"
RESULTS_DIR = "results"
CSV_DIR = "results/csv"
os.makedirs(CSV_DIR, exist_ok=True)


def run_error_analysis():
    print("=" * 65)
    print(" EXPERIMENTAL FRAMEWORK: MODEL ERROR ANALYSIS & MISCLASSIFICATION STUDY")
    print("=" * 65)

    test_path = os.path.join(DATA_DIR, "Testing")
    X_test, y_test, filenames, filepaths = load_split_into_memory(test_path)
    y_true = np.argmax(y_test, axis=1)

    model_path = os.path.join(SAVED_MODELS_DIR, "efficientnetb0.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(SAVED_MODELS_DIR, "baseline_cnn.h5")

    print(f" [MODEL] Loading model for error analysis: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    preds = model.predict(X_test, batch_size=32, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    confs = np.max(preds, axis=1)

    metrics = evaluate_predictions(y_test, preds)

    # 1. Full Prediction Results CSV
    pred_rows = []
    misclass_rows = []
    misclass_gallery = []

    for idx in range(len(X_test)):
        t_cls = CLASSES[y_true[idx]]
        p_cls = CLASSES[pred_classes[idx]]
        conf = float(confs[idx] * 100.0)
        is_correct = bool(y_true[idx] == pred_classes[idx])

        row_data = {
            "Filename": filenames[idx],
            "True_Class": t_cls,
            "Predicted_Class": p_cls,
            "Confidence_pct": round(conf, 2),
            "Is_Correct": is_correct,
            "Glioma_Prob": round(float(preds[idx, 0]) * 100.0, 2),
            "Meningioma_Prob": round(float(preds[idx, 1]) * 100.0, 2),
            "NoTumor_Prob": round(float(preds[idx, 2]) * 100.0, 2),
            "Pituitary_Prob": round(float(preds[idx, 3]) * 100.0, 2),
        }
        pred_rows.append(row_data)

        if not is_correct:
            misclass_rows.append(row_data)

            # Store top 10 misclassifications with Grad-CAM images for gallery
            if len(misclass_gallery) < 12:
                img_batch = np.expand_dims(X_test[idx], axis=0)
                heatmap = generate_gradcam_heatmap(model, img_batch, pred_index=pred_classes[idx])
                pil_orig = Image.fromarray(np.uint8(X_test[idx]))
                pil_overlay = heatmap_to_overlay(X_test[idx], heatmap)

                misclass_gallery.append({
                    "filename": filenames[idx],
                    "true_class": t_cls,
                    "predicted_class": p_cls,
                    "confidence": round(conf, 2),
                    "original_b64": pil_to_base64_uri(pil_orig),
                    "overlay_b64": pil_to_base64_uri(pil_overlay),
                })

    df_preds = pd.DataFrame(pred_rows)
    df_preds.to_csv(os.path.join(CSV_DIR, "prediction_results.csv"), index=False)
    print(f" [SAVED] prediction_results.csv -> {os.path.join(CSV_DIR, 'prediction_results.csv')}")

    df_misclass = pd.DataFrame(misclass_rows)
    df_misclass.to_csv(os.path.join(CSV_DIR, "misclassification_results.csv"), index=False)
    print(f" [SAVED] misclassification_results.csv -> {os.path.join(CSV_DIR, 'misclassification_results.csv')}")

    error_summary = {
        "total_test_images": len(X_test),
        "correct_predictions": int(np.sum(y_true == pred_classes)),
        "incorrect_predictions": int(np.sum(y_true != pred_classes)),
        "accuracy_pct": round(float(np.mean(y_true == pred_classes)) * 100.0, 2),
        "avg_confidence_correct_pct": metrics["avg_confidence_correct"],
        "avg_confidence_incorrect_pct": metrics["avg_confidence_incorrect"],
        "glioma_as_meningioma_errors": metrics["glioma_as_meningioma_errors"],
        "meningioma_as_glioma_errors": metrics["meningioma_as_glioma_errors"],
        "gallery": misclass_gallery,
    }

    with open(os.path.join(RESULTS_DIR, "error_analysis_summary.json"), "w") as f:
        json.dump(error_summary, f, indent=2)

    print(f"\n [SUMMARY] Total: {len(X_test)} | Correct: {error_summary['correct_predictions']} | Incorrect: {error_summary['incorrect_predictions']}")
    print(f"           Glioma -> Meningioma errors: {metrics['glioma_as_meningioma_errors']}")
    print(f"           Meningioma -> Glioma errors: {metrics['meningioma_as_glioma_errors']}")

    return error_summary


if __name__ == "__main__":
    run_error_analysis()
