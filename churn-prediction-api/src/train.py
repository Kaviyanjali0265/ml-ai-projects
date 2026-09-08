import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(__file__))

from evaluate import (
    compare_models,
    evaluate_model,
    get_shap_top_features,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_shap_summary,
    plot_shap_waterfall,
)
from preprocess import (
    clean_data,
    encode_features,
    load_data,
    load_feature_names,
    save_feature_names,
    scale_features,
    split_data,
)

DATA_PATH = "data/Telco_customer_churn.xlsx"
MODEL_OUTPUT_PATH = "models/model.pkl"
SCALER_OUTPUT_PATH = "models/scaler.pkl"
MLFLOW_EXPERIMENT = "churn-prediction"


def train_logistic_regression(X_train, y_train, X_val, y_val) -> tuple:
    with mlflow.start_run(run_name="logistic_regression"):
        params = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "random_state": 42}
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        results = evaluate_model("Logistic Regression", model, X_val, y_val)
        mlflow.log_metrics({"f1": results["f1"], "roc_auc": results["roc_auc"]})
        mlflow.sklearn.log_model(model, "model")

    return model, results


def train_decision_tree(X_train, y_train, X_val, y_val) -> tuple:
    with mlflow.start_run(run_name="decision_tree"):
        params = {"max_depth": 6, "min_samples_leaf": 20, "class_weight": "balanced", "random_state": 42}
        mlflow.log_params(params)

        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)

        results = evaluate_model("Decision Tree", model, X_val, y_val)
        mlflow.log_metrics({"f1": results["f1"], "roc_auc": results["roc_auc"]})
        mlflow.sklearn.log_model(model, "model")

    return model, results


def train_xgboost(X_train, y_train, X_val, y_val) -> tuple:
    with mlflow.start_run(run_name="xgboost"):
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        results = evaluate_model("XGBoost", model, X_val, y_val)
        mlflow.log_metrics({"f1": results["f1"], "roc_auc": results["roc_auc"]})
        mlflow.xgboost.log_model(model, "model")

    return model, results


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = encode_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    feature_names = X_train.columns.tolist()
    save_feature_names(feature_names)

    print("\nTraining models...")
    lr_model, lr_results = train_logistic_regression(X_train, y_train, X_val, y_val)
    dt_model, dt_results = train_decision_tree(X_train, y_train, X_val, y_val)
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_val, y_val)

    all_results = [lr_results, dt_results, xgb_results]
    compare_models(all_results)

    print("\nGenerating evaluation plots...")
    all_models = {
        "Logistic Regression": lr_model,
        "Decision Tree": dt_model,
        "XGBoost": xgb_model,
    }
    plot_roc_curves(all_models, X_test, y_test)

    for name, model in all_models.items():
        plot_confusion_matrix(name, model, X_test, y_test)

    print("\nGenerating SHAP plots for XGBoost...")
    plot_shap_summary(xgb_model, X_test.values, feature_names)
    plot_shap_waterfall(xgb_model, X_test.values, feature_names, sample_index=0)

    best = max(all_results, key=lambda x: x["f1"])
    best_model = all_models[best["model_name"]]
    print(f"\nBest model: {best['model_name']} with F1={best['f1']:.4f}")

    print("\nSaving best model and scaler...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")
    print(f"Scaler saved to {SCALER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
