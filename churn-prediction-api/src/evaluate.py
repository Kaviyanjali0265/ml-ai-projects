import os

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model_name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {auc:.4f}")

    return {"model_name": model_name, "f1": f1, "roc_auc": auc}


def plot_confusion_matrix(model_name: str, model, X_test, y_test, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["No Churn", "Churn"], cmap="Blues"
    )
    disp.ax_.set_title(f"Confusion Matrix - {model_name}")

    path = os.path.join(output_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {path}")

    return path


def plot_roc_curves(models: dict, X_test, y_test, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Models")
    plt.legend()

    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved to {path}")

    return path


def plot_shap_summary(model, X_test, feature_names: list, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

    path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {path}")

    return path


def plot_shap_waterfall(model, X_test, feature_names: list, sample_index: int = 0, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test)

    plt.figure()
    shap.plots.waterfall(explanation[sample_index], show=False)

    path = os.path.join(output_dir, f"shap_waterfall_sample_{sample_index}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"SHAP waterfall plot saved to {path}")

    return path


def compare_models(results: list[dict]):
    print(f"\n{'='*50}")
    print("Model Comparison")
    print(f"{'='*50}")
    print(f"{'Model':<30} {'F1 Score':<12} {'ROC-AUC'}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        print(f"{r['model_name']:<30} {r['f1']:<12.4f} {r['roc_auc']:.4f}")


def get_shap_top_features(model, input_array: np.ndarray, feature_names: list, top_n: int = 5) -> dict:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_array)

    feature_impacts = dict(zip(feature_names, shap_values[0].tolist()))
    sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    return dict(sorted_impacts[:top_n])
