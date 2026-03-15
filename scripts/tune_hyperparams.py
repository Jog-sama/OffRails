"""
Hyperparameter tuning for the XGBoost model using grid search with
cross-validation. Searches over tree depth, learning rate, estimator count,
and regularization parameters.

"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from build_features import get_feature_columns

warnings.filterwarnings("ignore")


PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
}

# smaller grid for faster iteration
PARAM_GRID_SMALL = {
    "n_estimators": [150, 300],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "min_child_weight": [1, 3],
    "gamma": [0, 0.1],
}


def run_tuning(data_dir: str, output_dir: str, fast: bool = True):
    """Run grid search CV on XGBoost."""
    os.makedirs(output_dir, exist_ok=True)

    train_feat = pd.read_parquet(os.path.join(data_dir, "train_features.parquet"))
    feat_cols = get_feature_columns(train_feat)

    X = train_feat[feat_cols]
    y = train_feat["label"].values

    # class imbalance weight
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    scale = n_neg / n_pos if n_pos > 0 else 1.0

    grid = PARAM_GRID_SMALL if fast else PARAM_GRID
    print(f"[Tuning] Grid size: {np.prod([len(v) for v in grid.values()])} combos")
    print(f"[Tuning] scale_pos_weight = {scale:.2f}")

    base_model = XGBClassifier(
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        base_model, grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )
    search.fit(X, y)

    print(f"\n[Tuning] Best F1 (macro): {search.best_score_:.4f}")
    print(f"[Tuning] Best params: {search.best_params_}")

    # save results
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    results_path = os.path.join(output_dir, "tuning_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"[SAVED] {results_path}")

    # save best params
    import json
    params_path = os.path.join(output_dir, "best_params.json")
    with open(params_path, "w") as f:
        json.dump(search.best_params_, f, indent=2)
    print(f"[SAVED] {params_path}")

    # plot top 10 configs
    top10 = results_df.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        range(len(top10)),
        top10["mean_test_score"],
        xerr=top10["std_test_score"],
        color="steelblue", capsize=3,
    )
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels([str(i+1) for i in range(len(top10))])
    ax.set_xlabel("F1 Macro (CV)")
    ax.set_ylabel("Rank")
    ax.set_title("Top 10 Hyperparameter Configurations")
    ax.invert_yaxis()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "tuning_top10.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {plot_path}")

    return search.best_params_


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="data/outputs")
    parser.add_argument("--full", action="store_true", help="Use full grid (slower)")
    args = parser.parse_args()

    best = run_tuning(args.data_dir, args.output_dir, fast=not args.full)
    print(f"\n[DONE] Best hyperparameters: {best}")


if __name__ == "__main__":
    main()
