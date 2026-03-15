"""
Focused experiment: Training Set Size Sensitivity + Noise Robustness Analysis.

Experiment 1 — Sensitivity Analysis:
    How does model performance degrade as we reduce training data?
    Trains XGBoost at 10%, 25%, 50%, 75%, 100% of training data
    and plots F1 vs. training set size.

Experiment 2 — Noise Robustness:
    How robust is the model to label noise (simulating proxy label errors)?
    Flips 5%, 10%, 15%, 20% of training labels randomly and measures
    degradation in test F1.

Both experiments are well-motivated for this project because:
  - Our labels are PROXY labels (heuristic-based), so understanding how
    noise in these labels affects model quality is directly relevant.
  - Knowing the data efficiency curve tells us whether we need more data
    or better features.
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from build_features import get_feature_columns
from model import ClassicalMLModel

warnings.filterwarnings("ignore")


def run_sensitivity_analysis(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    output_dir: str,
    fractions: list[float] = None,
    n_repeats: int = 3,
):
    """
    Train XGBoost at various training set sizes, measure test F1.
    Runs multiple seeds per fraction to estimate variance.
    """
    if fractions is None:
        fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

    feat_cols = get_feature_columns(train_feat)
    X_test = test_feat[feat_cols]
    y_test = test_feat["label"].values

    results = []
    print("\n" + "═" * 60)
    print("  EXPERIMENT 1: Training Set Size Sensitivity")
    print("═" * 60)

    for frac in fractions:
        f1_scores = []
        for seed in range(n_repeats):
            if frac < 1.0:
                sampled = train_feat.sample(frac=frac, random_state=seed)
            else:
                sampled = train_feat

            X_train = sampled[feat_cols]
            y_train = sampled["label"].values

            model = ClassicalMLModel(params={"n_estimators": 200, "max_depth": 5, "verbosity": 0})
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        n_samples = int(len(train_feat) * frac)

        results.append({
            "fraction": frac,
            "n_samples": n_samples,
            "mean_f1_macro": mean_f1,
            "std_f1_macro": std_f1,
        })
        print(f"  {frac*100:5.0f}% ({n_samples:6d} samples) → "
              f"F1 = {mean_f1:.4f} ± {std_f1:.4f}")

    results_df = pd.DataFrame(results)

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        results_df["n_samples"], results_df["mean_f1_macro"],
        yerr=results_df["std_f1_macro"],
        marker="o", linewidth=2, capsize=5, color="steelblue",
    )
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Experiment 1: Training Set Size vs. Model Performance")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "exp_sensitivity_analysis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")

    csv_path = os.path.join(output_dir, "exp_sensitivity_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    return results_df


def run_noise_robustness(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    output_dir: str,
    noise_rates: list[float] = None,
    n_repeats: int = 3,
):
    """
    Flip a fraction of training labels to simulate proxy label noise.
    Measure how test F1 degrades.
    """
    if noise_rates is None:
        noise_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

    feat_cols = get_feature_columns(train_feat)
    X_test = test_feat[feat_cols]
    y_test = test_feat["label"].values

    results = []
    print("\n" + "═" * 60)
    print("  EXPERIMENT 2: Label Noise Robustness")
    print("═" * 60)

    for noise in noise_rates:
        f1_scores = []
        for seed in range(n_repeats):
            rng = np.random.RandomState(seed)
            y_noisy = train_feat["label"].values.copy()

            if noise > 0:
                n_flip = int(len(y_noisy) * noise)
                flip_idx = rng.choice(len(y_noisy), size=n_flip, replace=False)
                y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

            X_train = train_feat[feat_cols]

            model = ClassicalMLModel(params={"n_estimators": 200, "max_depth": 5, "verbosity": 0})
            model.fit(X_train, y_noisy)

            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        results.append({
            "noise_rate": noise,
            "mean_f1_macro": mean_f1,
            "std_f1_macro": std_f1,
        })
        print(f"  Noise = {noise*100:5.1f}% → F1 = {mean_f1:.4f} ± {std_f1:.4f}")

    results_df = pd.DataFrame(results)

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        results_df["noise_rate"] * 100, results_df["mean_f1_macro"],
        yerr=results_df["std_f1_macro"],
        marker="s", linewidth=2, capsize=5, color="coral",
    )
    ax.set_xlabel("Label Noise Rate (%)")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Experiment 2: Proxy Label Noise vs. Model Performance")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "exp_noise_robustness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")

    csv_path = os.path.join(output_dir, "exp_noise_robustness_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="data/outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_feat = pd.read_parquet(os.path.join(args.data_dir, "train_features.parquet"))
    test_feat = pd.read_parquet(os.path.join(args.data_dir, "test_features.parquet"))

    sens_results = run_sensitivity_analysis(train_feat, test_feat, args.output_dir)
    noise_results = run_noise_robustness(train_feat, test_feat, args.output_dir)

    # combined summary
    print("\n" + "═" * 60)
    print("  EXPERIMENT SUMMARY & INTERPRETATION")
    print("═" * 60)

    # interpret sensitivity
    min_f1 = sens_results["mean_f1_macro"].iloc[0]
    max_f1 = sens_results["mean_f1_macro"].iloc[-1]
    improvement = max_f1 - min_f1
    print(f"\n  Sensitivity: F1 goes from {min_f1:.4f} (10%) to {max_f1:.4f} (100%)")
    print(f"  Improvement from 10x more data: {improvement:.4f}")

    if improvement < 0.02:
        print("  → Model is data-efficient; features capture signal well.")
    elif improvement < 0.05:
        print("  → Moderate benefit from more data; features are decent but more data helps.")
    else:
        print("  → Strong benefit from more data; consider collecting more labeled traces.")

    # interpret noise
    clean_f1 = noise_results["mean_f1_macro"].iloc[0]
    noisy_f1 = noise_results["mean_f1_macro"].iloc[-1]
    degradation = clean_f1 - noisy_f1
    print(f"\n  Noise Robustness: F1 drops from {clean_f1:.4f} (0%) to {noisy_f1:.4f} (25%)")
    print(f"  Total degradation: {degradation:.4f}")

    if degradation < 0.03:
        print("  → Model is highly robust to label noise. Proxy labels are probably fine.")
    elif degradation < 0.08:
        print("  → Moderate sensitivity to noise. Consider refining proxy labeling heuristics.")
    else:
        print("  → Significant noise sensitivity. Proxy labels may need manual verification.")

    print("\n[DONE] Experiments complete.")


if __name__ == "__main__":
    main()
