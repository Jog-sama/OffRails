"""
Runs the full pipeline: data download → feature extraction → training → evaluation.
"""

import argparse
import subprocess
import sys


def run_step(cmd: list[str], step_name: str):
    """Run a subprocess and handle errors."""
    print(f"\n{'═' * 60}")
    print(f"  STEP: {step_name}")
    print(f"{'═' * 60}\n")

    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=".",
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with code {result.returncode}")
        sys.exit(1)
    print(f"\n[OK] {step_name} complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Run full project pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "data", "features", "train", "evaluate",
                                 "experiment", "tune"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to load (for quick testing)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs for deep learning model")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "naive", "classical", "deep"])
    args = parser.parse_args()

    steps = {
        "data": lambda: run_step(
            ["scripts/make_dataset.py"] + (["--max_samples", str(args.max_samples)] if args.max_samples else []),
            "Data Download & Preprocessing"
        ),
        "features": lambda: run_step(
            ["scripts/build_features.py"],
            "Feature Extraction"
        ),
        "tune": lambda: run_step(
            ["scripts/tune_hyperparams.py"],
            "Hyperparameter Tuning"
        ),
        "train": lambda: run_step(
            ["scripts/train.py", "--model", args.model, "--epochs", str(args.epochs)],
            "Model Training"
        ),
        "evaluate": lambda: run_step(
            ["scripts/evaluate.py"],
            "Model Evaluation"
        ),
        "experiment": lambda: run_step(
            ["scripts/experiment.py"],
            "Experiments"
        ),
    }

    if args.step == "all":
        ordered = ["data", "features", "train", "evaluate", "experiment"]
        for step_name in ordered:
            steps[step_name]()
    else:
        steps[args.step]()

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
