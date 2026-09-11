import os
import sys

import mlflow
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

sys.path.append(os.path.dirname(__file__))

from dataset import MODEL_CHECKPOINT, load_imdb, tokenize_dataset
from evaluate import compute_metrics, evaluate_model, plot_confusion_matrix, plot_roc_curve

MODEL_OUTPUT_DIR = "models/sentiment-model"
MLFLOW_EXPERIMENT = "sentiment-analysis"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("Loading IMDB dataset...")
    train, val, test = load_imdb()
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    print("Tokenizing...")
    train_tok, val_tok, test_tok, tokenizer = tokenize_dataset(train, val, test)

    print("Loading DistilBERT...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    with mlflow.start_run(run_name="distilbert-imdb"):
        mlflow.log_params({
            "model": MODEL_CHECKPOINT,
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "train_size": len(train),
            "max_length": 256,
        })

        print("\nFine-tuning DistilBERT...")
        trainer.train()

        print("\nEvaluating on test set...")
        predictions_output = trainer.predict(test_tok)
        logits = predictions_output.predictions
        labels = predictions_output.label_ids
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        preds = np.argmax(logits, axis=-1)

        results = evaluate_model("DistilBERT-IMDB", labels, preds, probs)
        mlflow.log_metrics({"f1": results["f1"], "roc_auc": results["roc_auc"]})

        plot_confusion_matrix("DistilBERT", labels, preds)
        plot_roc_curve("DistilBERT", labels, probs)

    print(f"\nSaving model to {MODEL_OUTPUT_DIR}...")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
