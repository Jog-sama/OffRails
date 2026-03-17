"""
Main entry point for the Agent Trace Anomaly Detection project.

Modes:
  - pipeline:  Run the full ML pipeline (data → features → train → evaluate)
  - inference: Run inference on a single trace JSON file
  - demo:      Interactive demo that accepts a pasted trace and returns prediction

Usage:
    python main.py pipeline                          # full pipeline
    python main.py pipeline --max_samples 5000       # quick test run
    python main.py inference --trace traces/ex.json  # single trace
    python main.py demo                              # interactive mode
"""

import argparse
import json
import os
import sys

# add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))


def run_pipeline(args):
    """Run the full training pipeline via setup.py."""
    from setup import main as setup_main
    sys.argv = ["setup.py", "--step", "all"]
    if args.max_samples:
        sys.argv += ["--max_samples", str(args.max_samples)]
    setup_main()


def run_inference(args):
    """Run inference on a single trace file."""
    from scripts.inference import TraceAnomalyDetector

    with open(args.trace) as f:
        conversations = json.load(f)

    detector = TraceAnomalyDetector(
        model_dir=args.model_dir,
        model_type=args.model_type,
    )
    result = detector.predict(conversations)
    print(json.dumps(result, indent=2, default=str))


def run_demo(args):
    """Interactive demo: paste a trace, get a prediction."""
    from scripts.inference import TraceAnomalyDetector

    detector = TraceAnomalyDetector(
        model_dir=args.model_dir,
        model_type=args.model_type,
    )

    print("=" * 50)
    print("  Agent Trace Anomaly Detector — Demo Mode")
    print("=" * 50)
    print("Paste a JSON conversation trace (list of message dicts),")
    print("then press Enter twice to submit. Type 'quit' to exit.\n")

    while True:
        print("─" * 50)
        lines = []
        try:
            while True:
                line = input()
                if line.strip().lower() == "quit":
                    print("Bye!")
                    return
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
        except EOFError:
            break

        text = "\n".join(lines).strip()
        if not text:
            continue

        try:
            conversations = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            continue

        result = detector.predict(conversations)

        status = "ANOMALOUS" if result["is_anomalous"] else "NORMAL"
        conf = result["confidence"]
        print(f"\n  Prediction: {status} (confidence: {conf:.2%})")

        if result["anomaly_signals"]:
            print("  Signals:")
            for sig in result["anomaly_signals"]:
                print(f"    • {sig}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Agent Trace Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pipeline
    pipe_parser = subparsers.add_parser("pipeline", help="Run full ML pipeline")
    pipe_parser.add_argument("--max_samples", type=int, default=None)

    # inference
    inf_parser = subparsers.add_parser("inference", help="Run inference on a trace")
    inf_parser.add_argument("--trace", type=str, required=True, help="Path to trace JSON")
    inf_parser.add_argument("--model_dir", type=str, default="models")
    inf_parser.add_argument("--model_type", type=str, default="xgboost",
                            choices=["xgboost", "distilbert"])

    # demo
    demo_parser = subparsers.add_parser("demo", help="Interactive demo")
    demo_parser.add_argument("--model_dir", type=str, default="models")
    demo_parser.add_argument("--model_type", type=str, default="xgboost",
                             choices=["xgboost", "distilbert"])

    args = parser.parse_args()

    if args.command == "pipeline":
        run_pipeline(args)
    elif args.command == "inference":
        run_inference(args)
    elif args.command == "demo":
        run_demo(args)


if __name__ == "__main__":
    main()
