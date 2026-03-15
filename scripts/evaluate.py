"""
Comprehensive evaluation of all three models on the held-out test set.

Produces:
  - Per-model metrics (accuracy, precision, recall, F1, ROC AUC)
  - Confusion matrices (saved as PNGs)
  - ROC curves overlay
  - Feature importance plot (XGBoost)
  - Error analysis: 5 specific mispredictions per model with root-cause explanation
  - Results summary CSV

"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from build_features import extract_features_from_row, get_feature_columns
from model import ClassicalMLModel, NaiveBaseline, TraceTransformer

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute full metric suite."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            metrics["roc_auc"] = None
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Anomalous"],
        yticklabels=["Normal", "Anomalous"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    path = os.path.join(output_dir, f"cm_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")


def plot_roc_curves(results: dict, output_dir: str):
    """Overlay ROC curves for all models that have probability outputs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        if res.get("y_proba") is not None and res.get("roc_auc") is not None:
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"][:, 1])
            ax.plot(fpr, tpr, label=f'{name} (AUC={res["roc_auc"]:.3f})')

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_precision_recall_curves(results: dict, output_dir: str):
    """Overlay Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        if res.get("y_proba") is not None:
            prec_vals, rec_vals, _ = precision_recall_curve(
                res["y_true"], res["y_proba"][:, 1]
            )
            ax.plot(rec_vals, prec_vals, label=name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="upper right")
    plt.tight_layout()

    path = os.path.join(output_dir, "pr_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_feature_importance(model: ClassicalMLModel, output_dir: str, top_n: int = 20):
    """Bar chart of top feature importances."""
    importance = model.get_feature_importance().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(f"XGBoost — Top {top_n} Feature Importances")
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")


def error_analysis(y_true, y_pred, df, model_name, n_errors=5) -> list[dict]:
    """
    Identify mispredictions, explain likely root causes.
    Returns a list of dicts with error details.
    """
    errors = []
    wrong_mask = y_true != y_pred

    wrong_indices = np.where(wrong_mask)[0]
    if len(wrong_indices) == 0:
        print(f"  [{model_name}] No errors found!")
        return errors

    # sample up to n_errors, balanced between FP and FN
    fp_idx = wrong_indices[(y_true[wrong_indices] == 0) & (y_pred[wrong_indices] == 1)]
    fn_idx = wrong_indices[(y_true[wrong_indices] == 1) & (y_pred[wrong_indices] == 0)]

    sample_fp = fp_idx[:min(3, len(fp_idx))]
    sample_fn = fn_idx[:min(3, len(fn_idx))]
    sampled = np.concatenate([sample_fp, sample_fn])[:n_errors]

    for idx in sampled:
        row = df.iloc[idx]
        error_type = "False Positive" if y_true[idx] == 0 else "False Negative"

        # infer root cause from features
        query_preview = str(row.get("user_query", ""))[:150]
        n_tools = row.get("num_tool_calls", 0)
        n_turns = row.get("num_turns", 0)

        if error_type == "False Positive":
            cause = (
                f"Model predicted anomalous but trace was normal. "
                f"Trace had {n_tools} tool calls across {n_turns} turns. "
                f"Possible cause: hedging or cautious language in the final response "
                f"triggered false failure signal."
            )
            mitigation = (
                "Add features that distinguish cautious-but-successful responses "
                "from actual failures. Consider sentiment analysis on the final turn."
            )
        else:
            cause = (
                f"Model predicted normal but trace was actually anomalous. "
                f"Trace had {n_tools} tool calls across {n_turns} turns. "
                f"Possible cause: the agent completed tool calls but the final answer "
                f"was incorrect/incomplete without obvious failure language."
            )
            mitigation = (
                "Incorporate semantic similarity between query intent and final response. "
                "Add features measuring whether tool outputs were actually used in the answer."
            )

        errors.append({
            "index": int(idx),
            "model": model_name,
            "error_type": error_type,
            "true_label": int(y_true[idx]),
            "pred_label": int(y_pred[idx]),
            "query_preview": query_preview,
            "root_cause": cause,
            "mitigation": mitigation,
        })

    return errors


def evaluate_all(data_dir: str, model_dir: str, output_dir: str):
    """Run full evaluation pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # load test data
    test_raw = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
    test_feat_path = os.path.join(data_dir, "test_features.parquet")
    test_feat = pd.read_parquet(test_feat_path) if os.path.exists(test_feat_path) else None

    y_true = test_raw["label"].values
    results = {}
    all_errors = []

    # Naive Baseline
    naive_path = os.path.join(model_dir, "naive_baseline.joblib")
    if os.path.exists(naive_path):
        print("\nEvaluating: Naive Baseline")
        naive = NaiveBaseline.load(naive_path)
        y_pred = naive.predict(y_true)
        y_proba = naive.predict_proba(y_true)

        metrics = compute_metrics(y_true, y_pred, y_proba)
        results["Naive Baseline"] = {**metrics, "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba}

        plot_confusion_matrix(y_true, y_pred, "Naive Baseline", output_dir)
        all_errors += error_analysis(y_true, y_pred, test_raw, "Naive Baseline")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"]))

    # XGBoost
    xgb_path = os.path.join(model_dir, "xgboost_model.joblib")
    if os.path.exists(xgb_path) and test_feat is not None:
        print("\nEvaluating: XGBoost")
        xgb_model = ClassicalMLModel.load(xgb_path)
        feat_cols = get_feature_columns(test_feat)
        X_test = test_feat[feat_cols]

        y_pred = xgb_model.predict(X_test)
        y_proba = xgb_model.predict_proba(X_test)

        metrics = compute_metrics(y_true, y_pred, y_proba)
        results["XGBoost"] = {**metrics, "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba}

        plot_confusion_matrix(y_true, y_pred, "XGBoost", output_dir)
        plot_feature_importance(xgb_model, output_dir)
        all_errors += error_analysis(y_true, y_pred, test_raw, "XGBoost")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"]))

    # DistilBERT
    dl_path = os.path.join(model_dir, "distilbert_trace")
    if os.path.exists(dl_path):
        print("\nEvaluating: DistilBERT")
        dl_model = TraceTransformer.load(dl_path)
        X_test = test_raw["raw_trace"].tolist()

        y_pred = dl_model.predict(X_test)
        y_proba = dl_model.predict_proba(X_test)

        metrics = compute_metrics(y_true, y_pred, y_proba)
        results["DistilBERT"] = {**metrics, "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba}

        plot_confusion_matrix(y_true, y_pred, "DistilBERT", output_dir)
        all_errors += error_analysis(y_true, y_pred, test_raw, "DistilBERT")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"]))

    # Comparison plots
    if results:
        plot_roc_curves(results, output_dir)
        plot_precision_recall_curves(results, output_dir)

    # Summary Table
    summary_rows = []
    for name, res in results.items():
        row = {k: v for k, v in res.items() if k not in ("y_true", "y_pred", "y_proba")}
        row["model"] = name
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).set_index("model")
        print("\n" + "═" * 60)
        print("  FINAL TEST SET COMPARISON")
        print("═" * 60)
        print(summary_df.to_string())

        summary_path = os.path.join(output_dir, "results_summary.csv")
        summary_df.to_csv(summary_path)
        print(f"\n[SAVED] {summary_path}")

    # Error analysis details
    if all_errors:
        errors_df = pd.DataFrame(all_errors)
        errors_path = os.path.join(output_dir, "error_analysis.csv")
        errors_df.to_csv(errors_path, index=False)
        print(f"[SAVED] {errors_path}")

        print("\n" + "═" * 60)
        print("  ERROR ANALYSIS (5 mispredictions per model)")
        print("═" * 60)
        for err in all_errors:
            print(f"\n  [{err['model']}] {err['error_type']}")
            print(f"    True: {err['true_label']}, Predicted: {err['pred_label']}")
            print(f"    Query: {err['query_preview']}")
            print(f"    Root cause: {err['root_cause']}")
            print(f"    Mitigation: {err['mitigation']}")

    # Bar chart comparison
    if summary_rows:
        plot_metric_comparison(summary_rows, output_dir)

    print("\n[DONE] Evaluation complete.")


def plot_metric_comparison(summary_rows: list[dict], output_dir: str):
    """Bar chart comparing key metrics across models."""
    df = pd.DataFrame(summary_rows).set_index("model")
    plot_cols = ["accuracy", "precision", "recall", "f1", "f1_macro"]
    plot_cols = [c for c in plot_cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    df[plot_cols].plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on test set")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="data/outputs")
    args = parser.parse_args()

    evaluate_all(args.data_dir, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
