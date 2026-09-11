import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

OUTPUT_DIR = "models"


def evaluate_model(name: str, model, loader, criterion, device) -> dict:
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            probs = model(X_batch)
            preds = (probs >= 0.5).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    print(f"\n{name}")
    print(classification_report(all_labels, all_preds, target_names=["No Disease", "Disease"]))
    print(f"ROC-AUC: {roc_auc:.4f}")

    return {"model_name": name, "f1": f1, "roc_auc": roc_auc, "probs": all_probs, "labels": all_labels}


def plot_training_history(train_losses: list, val_losses: list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
    plt.close()
    print("Saved training_history.png")


def plot_confusion_matrix(name: str, labels: np.ndarray, preds: np.ndarray):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    disp.plot(cmap="Blues")
    disp.ax_.set_title(f"Confusion Matrix - {name}")
    filename = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved {filename}")


def plot_roc_curve(name: str, labels: np.ndarray, probs: np.ndarray):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    RocCurveDisplay.from_predictions(labels, probs, name=name)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close()
    print("Saved roc_curve.png")
