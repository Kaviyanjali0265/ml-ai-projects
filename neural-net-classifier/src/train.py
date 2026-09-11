import os
import sys

import joblib
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))

from dataset import LoanDataset
from evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
)
from model import LoanNet
from preprocess import (
    clean_data,
    encode_features,
    load_data,
    save_feature_names,
    scale_features,
    split_data,
)

DATA_PATH = "data/train.csv"
MODEL_OUTPUT_PATH = "models/model.pt"
SCALER_OUTPUT_PATH = "models/scaler.pkl"
MLFLOW_EXPERIMENT = "loan-default-net"

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 10


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    train_dataset = LoanDataset(X_train.values, y_train)
    val_dataset = LoanDataset(X_val.values, y_val)
    test_dataset = LoanDataset(X_test.values, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    input_dim = X_train.shape[1]
    model = LoanNet(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with mlflow.start_run(run_name="loan_net"):
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "patience": PATIENCE,
            "architecture": "64-32-1",
            "dropout": 0.3,
        })

        print("\nTraining...")
        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(best_weights)

        results = evaluate_model("LoanNet", model, test_loader, criterion, device)
        mlflow.log_metrics({"f1": results["f1"], "roc_auc": results["roc_auc"]})
        mlflow.pytorch.log_model(model, "model", serialization_format="pickle")

        plot_training_history(train_losses, val_losses)
        plot_confusion_matrix("LoanNet", results["labels"], (results["probs"] >= 0.5).astype(int))
        plot_roc_curve("LoanNet", results["labels"], results["probs"])

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")
    print(f"Scaler saved to {SCALER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
