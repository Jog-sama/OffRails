"""
train.py — Orchestrates training of all three models.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model import ClassicalMLModel, NaiveBaseline, TraceTransformer
from build_features import get_feature_columns

warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(data_dir: str):
    splits = {}
    for name in ["train", "val", "test"]:
        raw_path = os.path.join(data_dir, f"{name}.parquet")
        feat_path = os.path.join(data_dir, f"{name}_features.parquet")
        if os.path.exists(raw_path):
            splits[f"{name}_raw"] = pd.read_parquet(raw_path)
        if os.path.exists(feat_path):
            splits[f"{name}_feat"] = pd.read_parquet(feat_path)
    return splits


def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    auc = "N/A"
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            auc = f"{roc_auc_score(y_true, y_proba[:, 1]):.4f}"
        except Exception:
            auc = "N/A"

    print(f"\n{'=' * 60}")
    print(f"  {model_name} — Validation Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1 (binary): {f1:.4f}")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  ROC AUC:     {auc}")

    # safe classification_report: use labels= to avoid crash on single class
    all_labels = sorted(set(y_true) | set(y_pred))
    names_map = {0: "Normal", 1: "Anomalous"}
    target_names = [names_map.get(l, str(l)) for l in all_labels]
    print(f"\n{classification_report(y_true, y_pred, labels=all_labels, target_names=target_names, zero_division=0)}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "f1_macro": f1_macro}


def train_naive(splits, model_dir):
    print("\n" + "─" * 60)
    print("  TRAINING: Naive Baseline (Majority Class)")
    print("─" * 60)

    train_df = splits.get("train_feat", splits.get("train_raw"))
    val_df = splits.get("val_feat", splits.get("val_raw"))

    model = NaiveBaseline()
    model.fit(None, train_df["label"].values)

    y_val = val_df["label"].values
    y_pred = model.predict(y_val)
    y_proba = model.predict_proba(y_val)

    metrics = evaluate_model(y_val, y_pred, y_proba, "Naive Baseline")

    save_path = os.path.join(model_dir, "naive_baseline.joblib")
    model.save(save_path)
    print(f"[SAVED] {save_path}")
    return metrics


def train_classical(splits, model_dir):
    print("\n" + "─" * 60)
    print("  TRAINING: XGBoost (Classical ML)")
    print("─" * 60)

    train_feat = splits["train_feat"]
    val_feat = splits["val_feat"]
    feat_cols = get_feature_columns(train_feat)

    X_train = train_feat[feat_cols]
    y_train = train_feat["label"].values
    X_val = val_feat[feat_cols]
    y_val = val_feat["label"].values

    # check we have both classes
    if len(np.unique(y_train)) < 2:
        print("[ERROR] Training data has only one class. Cannot train XGBoost.")
        print(f"        Label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "f1_macro": 0}

    model = ClassicalMLModel()
    model.fit(X_train, y_train, X_val, y_val)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    metrics = evaluate_model(y_val, y_pred, y_proba, "XGBoost")

    importance = model.get_feature_importance()
    print("\nTop 15 features:")
    for feat_name, score in importance.head(15).items():
        print(f"  {feat_name:40s} {score:.4f}")

    save_path = os.path.join(model_dir, "xgboost_model.joblib")
    model.save(save_path)
    print(f"\n[SAVED] {save_path}")
    return metrics


def train_deep(splits, model_dir, num_epochs=3, batch_size=16, lr=2e-5):
    print("\n" + "─" * 60)
    print("  TRAINING: DistilBERT (Deep Learning)")
    print("─" * 60)

    train_raw = splits["train_raw"]
    val_raw = splits["val_raw"]

    X_train = train_raw["raw_trace"].tolist()
    y_train = train_raw["label"].tolist()
    X_val = val_raw["raw_trace"].tolist()
    y_val = val_raw["label"].tolist()

    if len(set(y_train)) < 2:
        print("[ERROR] Training data has only one class. Cannot train DistilBERT.")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "f1_macro": 0}

    model = TraceTransformer(
        model_name="distilbert-base-uncased",
        max_length=512,
        batch_size=batch_size,
        learning_rate=lr,
        num_epochs=num_epochs,
    )
    model.fit(X_train, y_train, X_val, y_val)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    y_val_arr = np.array(y_val)

    metrics = evaluate_model(y_val_arr, y_pred, y_proba, "DistilBERT")

    save_path = os.path.join(model_dir, "distilbert_trace")
    model.save(save_path)
    print(f"\n[SAVED] {save_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "naive", "classical", "deep"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    splits = load_data(args.data_dir)
    results = {}

    if args.model in ("all", "naive"):
        results["naive"] = train_naive(splits, args.model_dir)

    if args.model in ("all", "classical"):
        if "train_feat" not in splits:
            print("[ERROR] Feature files not found. Run build_features.py first.")
            sys.exit(1)
        results["classical"] = train_classical(splits, args.model_dir)

    if args.model in ("all", "deep"):
        if "train_raw" not in splits:
            print("[ERROR] Raw data files not found. Run make_dataset.py first.")
            sys.exit(1)
        results["deep"] = train_deep(splits, args.model_dir,
                                      num_epochs=args.epochs,
                                      batch_size=args.batch_size, lr=args.lr)

    if len(results) > 1:
        print("\n" + "═" * 60)
        print("  MODEL COMPARISON SUMMARY")
        print("═" * 60)
        summary = pd.DataFrame(results).T
        print(summary.to_string())

    print("\n[DONE] All training complete.")


if __name__ == "__main__":
    main()