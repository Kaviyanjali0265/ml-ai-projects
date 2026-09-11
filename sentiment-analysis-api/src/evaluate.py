import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

OUTPUT_DIR = "models"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions)
    return {"f1": f1}


def evaluate_model(name: str, labels: np.ndarray, predictions: np.ndarray, probs: np.ndarray) -> dict:
    f1 = f1_score(labels, predictions)
    roc_auc = roc_auc_score(labels, probs)

    print(f"\n{name}")
    print(classification_report(labels, predictions, target_names=["Negative", "Positive"]))
    print(f"ROC-AUC: {roc_auc:.4f}")

    return {"model_name": name, "f1": f1, "roc_auc": roc_auc}


def plot_confusion_matrix(name: str, labels: np.ndarray, predictions: np.ndarray):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Purples")
    disp.ax_.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    print("Saved confusion_matrix.png")


def plot_roc_curve(name: str, labels: np.ndarray, probs: np.ndarray):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    RocCurveDisplay.from_predictions(labels, probs, name=name)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close()
    print("Saved roc_curve.png")
