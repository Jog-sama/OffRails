"""
Inference API for the deployed model. This module provides a clean interface
that the FastAPI backend can import and call.

Supports both XGBoost (on handcrafted features) and DistilBERT (on raw text).
The default deployed model is XGBoost since it's faster and doesn't need GPU.

Usage (from FastAPI):
    from scripts.inference import TraceAnomalyDetector
    detector = TraceAnomalyDetector(model_dir="models", model_type="xgboost")
    result = detector.predict(conversation_json)
"""

import argparse
import json
import os
import sys
from typing import Any
 
import numpy as np
import pandas as pd
 
from build_features import extract_features_from_row, get_feature_columns
from make_dataset import extract_raw_trace_text, parse_conversation
from model import ClassicalMLModel, TraceTransformer
 
 
class TraceAnomalyDetector:
    """
    Production inference wrapper. Accepts raw agent conversation traces
    and returns anomaly predictions with confidence scores.
    """
 
    def __init__(self, model_dir: str = "models", model_type: str = "xgboost"):
        """
        Load a trained model for inference.
 
        Args:
            model_dir: directory containing saved models
            model_type: "xgboost" or "distilbert"
        """
        self.model_type = model_type
 
        if model_type == "xgboost":
            path = os.path.join(model_dir, "xgboost_model.joblib")
            self.model = ClassicalMLModel.load(path)
            self.feature_cols = self.model.feature_names
            print(f"[Inference] Loaded XGBoost model from {path}")
        elif model_type == "distilbert":
            path = os.path.join(model_dir, "distilbert_trace")
            self.model = TraceTransformer.load(path)
            print(f"[Inference] Loaded DistilBERT model from {path}")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
 
    def predict(self, conversations: list[dict]) -> dict[str, Any]:
        """
        Run anomaly detection on a single agent trace.
 
        Args:
            conversations: list of message dicts with 'from'/'role' and 'value'/'content' keys.
                           This is the raw conversation in ShareGPT/ToolBench format.
 
        Returns:
            dict with:
                - is_anomalous: bool
                - confidence: float (probability of anomaly)
                - label: int (0=normal, 1=anomalous)
                - anomaly_signals: list of strings explaining why it might be anomalous
        """
        if self.model_type == "xgboost":
            return self._predict_xgboost(conversations)
        else:
            return self._predict_distilbert(conversations)
 
    def _predict_xgboost(self, conversations: list[dict]) -> dict[str, Any]:
        """XGBoost inference using handcrafted features."""
        # build a fake row to reuse feature extraction
        raw_text = extract_raw_trace_text(conversations)
        parsed = parse_conversation(conversations)
 
        row = pd.Series({
            "id": "inference",
            "user_query": parsed["user_query"],
            "num_turns": len(parsed["turns"]),
            "num_tool_calls": len(parsed["tool_calls"]),
            "num_observations": len(parsed["observations"]),
            "num_assistant_turns": len(parsed["assistant_turns"]),
            "raw_trace": raw_text,
            "conversations_json": json.dumps(conversations),
            "label": 0,
        })
 
        features = extract_features_from_row(row)
        feat_df = pd.DataFrame([features])
 
        # ensure column order matches training
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in feat_df.columns:
                    feat_df[col] = 0
            feat_df = feat_df[self.feature_cols]
 
        proba = self.model.predict_proba(feat_df)[0]
        pred = int(np.argmax(proba))
        confidence = float(proba[1])  # probability of anomalous
 
        # generate human-readable anomaly signals
        signals = self._generate_signals(features)
 
        return {
            "is_anomalous": pred == 1,
            "confidence": confidence,
            "label": pred,
            "anomaly_signals": signals,
            "features": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in features.items() if k not in ("id", "label")},
        }
 
    def _predict_distilbert(self, conversations: list[dict]) -> dict[str, Any]:
        """DistilBERT inference on raw trace text."""
        raw_text = extract_raw_trace_text(conversations)
        proba = self.model.predict_proba([raw_text])[0]
        pred = int(np.argmax(proba))
        confidence = float(proba[1])
 
        return {
            "is_anomalous": pred == 1,
            "confidence": confidence,
            "label": pred,
            "anomaly_signals": [],  # no handcrafted features for signal extraction
        }
 
    def _generate_signals(self, features: dict) -> list[str]:
        """
        Generate human-readable explanations of anomaly signals.
        """
        signals = []
 
        if features.get("num_tool_calls", 0) == 0:
            signals.append("No tool calls were made during the trace.")
 
        if features.get("max_consecutive_same_tool", 0) >= 3:
            signals.append(
                f"Circular behavior detected: same tool called "
                f"{features['max_consecutive_same_tool']} times consecutively."
            )
 
        if features.get("num_repeated_exact_calls", 0) > 3:
            signals.append(
                f"High tool repetition: {features['num_repeated_exact_calls']} "
                f"duplicate tool calls."
            )
 
        if features.get("tool_diversity_ratio", 1.0) < 0.3 and features.get("num_tool_calls", 0) > 2:
            signals.append(
                f"Low tool diversity: ratio = {features['tool_diversity_ratio']:.2f}. "
                f"Agent may be stuck in a loop."
            )
 
        if features.get("last_turn_apology_keywords", 0) >= 2:
            signals.append("Final response contains multiple apology/failure phrases.")
 
        if features.get("num_error_observations", 0) > 2:
            signals.append(
                f"{features['num_error_observations']} tool responses contained error indicators."
            )
 
        if features.get("give_up_keyword_count", 0) > 0:
            signals.append("Agent used language suggesting it gave up on the task.")
 
        if features.get("num_empty_observations", 0) > 1:
            signals.append(
                f"{features['num_empty_observations']} tool calls returned empty/minimal responses."
            )
 
        return signals
 
    def predict_batch(self, traces: list[list[dict]]) -> list[dict]:
        """Run inference on multiple traces."""
        return [self.predict(trace) for trace in traces]
 
 
def main():
    """CLI for quick inference testing."""
    parser = argparse.ArgumentParser(description="Run inference on a trace")
    parser.add_argument("--trace", type=str, required=True, help="Path to JSON trace file")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model_type", type=str, default="xgboost", choices=["xgboost", "distilbert"])
    args = parser.parse_args()
 
    with open(args.trace) as f:
        conversations = json.load(f)
 
    detector = TraceAnomalyDetector(model_dir=args.model_dir, model_type=args.model_type)
    result = detector.predict(conversations)
 
    print(json.dumps(result, indent=2, default=str))
 
 
if __name__ == "__main__":
    main()