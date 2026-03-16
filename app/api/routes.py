"""
FastAPI routes for Agent Trace Anomaly Detection.

All ML logic lives in scripts/inference.py (partner's code).
This file only handles HTTP ↔ inference translation.

Endpoints
---------
GET  /health                 – service health & loaded model info
POST /models/load            – load a model (xgboost or distilbert)
POST /predict                – predict anomaly for a single trace
POST /predict/batch          – predict anomalies for multiple traces
POST /predict/compare        – run both models on same trace, compare
POST /pipeline/train         – trigger the full training pipeline
GET  /pipeline/status        – check if models dir has trained models
"""

from __future__ import annotations

import json
import os
import subprocess
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.schemas import (
    PredictRequest,
    PredictBatchRequest,
    PredictResponse,
    PredictBatchResponse,
    CompareResponse,
    ModelLoadRequest,
    HealthResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── In-memory state ──────────────────────────────────────────────────────────

_state: dict[str, Any] = {
    "detector": None,          # TraceAnomalyDetector instance
    "model_type": None,        # "xgboost" or "distilbert"
    "model_dir": "models",     # path to saved models
}


def _get_detector():
    """Return the loaded detector or raise 409."""
    if _state["detector"] is None:
        raise HTTPException(
            status_code=409,
            detail="No model loaded. POST /models/load first.",
        )
    return _state["detector"]


def _check_available_models(model_dir: str) -> list[str]:
    """Check which trained models exist on disk."""
    available = []
    if os.path.exists(os.path.join(model_dir, "xgboost_model.joblib")):
        available.append("xgboost")
    if os.path.exists(os.path.join(model_dir, "distilbert_trace", "trace_config.json")):
        available.append("distilbert")
    if os.path.exists(os.path.join(model_dir, "naive_baseline.joblib")):
        available.append("naive_baseline")
    return available


# ── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    model_dir = _state["model_dir"]
    return HealthResponse(
        status="ok",
        loaded_model=_state["model_type"],
        available_models=_check_available_models(model_dir),
        model_dir=model_dir,
    )


# ── Model Loading ───────────────────────────────────────────────────────────

@router.post("/models/load", tags=["Models"])
def load_model(req: ModelLoadRequest):
    """
    Load a trained model into memory for inference.
    Models must be trained first via the pipeline (setup.py).
    """
    from scripts.inference import TraceAnomalyDetector

    available = _check_available_models(req.model_dir)
    if req.model_type not in available:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{req.model_type}' not found in '{req.model_dir}/'. "
                f"Available models: {available}. "
                f"Run the training pipeline first: python setup.py"
            ),
        )

    try:
        detector = TraceAnomalyDetector(
            model_dir=req.model_dir,
            model_type=req.model_type,
        )
        _state["detector"] = detector
        _state["model_type"] = req.model_type
        _state["model_dir"] = req.model_dir

        return {
            "message": f"Model '{req.model_type}' loaded successfully",
            "model_type": req.model_type,
            "model_dir": req.model_dir,
        }
    except Exception as exc:
        logger.exception("Failed to load model")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")


# ── Prediction ───────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict whether a single agent trace is anomalous.

    Pass conversations in ShareGPT/ToolBench format:
    [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}, ...]
    """
    detector = _get_detector()

    try:
        result = detector.predict(req.conversations)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return PredictResponse(
        is_anomalous=result["is_anomalous"],
        confidence=result["confidence"],
        label=result["label"],
        anomaly_signals=result.get("anomaly_signals", []),
        model_used=_state["model_type"],
        features=result.get("features"),
    )


@router.post("/predict/batch", response_model=PredictBatchResponse, tags=["Prediction"])
def predict_batch(req: PredictBatchRequest):
    """Predict anomalies for multiple traces at once."""
    detector = _get_detector()

    try:
        results = detector.predict_batch(req.traces)
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}")

    predictions = [
        PredictResponse(
            is_anomalous=r["is_anomalous"],
            confidence=r["confidence"],
            label=r["label"],
            anomaly_signals=r.get("anomaly_signals", []),
            model_used=_state["model_type"],
            features=r.get("features"),
        )
        for r in results
    ]

    return PredictBatchResponse(
        predictions=predictions,
        anomaly_count=sum(1 for p in predictions if p.is_anomalous),
        total=len(predictions),
    )


@router.post("/predict/compare", response_model=CompareResponse, tags=["Prediction"])
def predict_compare(req: PredictRequest):
    """
    Run both XGBoost and DistilBERT on the same trace and compare results.
    Both models must be trained and available in the models directory.
    """
    from scripts.inference import TraceAnomalyDetector

    model_dir = _state["model_dir"]
    available = _check_available_models(model_dir)
    results = {}

    for model_type in ["xgboost", "distilbert"]:
        if model_type in available:
            try:
                det = TraceAnomalyDetector(model_dir=model_dir, model_type=model_type)
                r = det.predict(req.conversations)
                results[model_type] = PredictResponse(
                    is_anomalous=r["is_anomalous"],
                    confidence=r["confidence"],
                    label=r["label"],
                    anomaly_signals=r.get("anomaly_signals", []),
                    model_used=model_type,
                    features=r.get("features"),
                )
            except Exception as exc:
                logger.warning("Compare: %s failed: %s", model_type, exc)

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No trained models found in '{model_dir}/'. Run the training pipeline first.",
        )

    xgb = results.get("xgboost")
    bert = results.get("distilbert")
    agreement = True
    if xgb and bert:
        agreement = xgb.label == bert.label

    return CompareResponse(
        xgboost=xgb,
        distilbert=bert,
        agreement=agreement,
    )


# ── Training Pipeline ────────────────────────────────────────────────────────

@router.post("/pipeline/train", tags=["Pipeline"])
def trigger_training(
    max_samples: int | None = Query(None, ge=100, description="Cap on dataset rows"),
    model: str = Query("all", description="Which model to train: all, naive, classical, deep"),
):
    """
    Trigger the full training pipeline (setup.py).

    This runs data download → feature extraction → model training.
    May take several minutes depending on dataset size and model choice.
    """
    cmd = ["python", "setup.py"]
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    if model != "all":
        cmd.extend(["--step", "train", "--model", model])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,  # 30 min max
        )
        return {
            "message": "Training pipeline completed" if result.returncode == 0 else "Pipeline failed",
            "returncode": result.returncode,
            "stdout": result.stdout[-3000:] if result.stdout else "",  # last 3000 chars
            "stderr": result.stderr[-1000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Training pipeline timed out (30 min limit)")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="setup.py not found. Make sure you're running from the OffRails project root.",
        )


@router.get("/pipeline/status", tags=["Pipeline"])
def pipeline_status():
    """Check what trained models and data files are available."""
    model_dir = _state["model_dir"]
    data_dir = "data/processed"

    data_files = {}
    if os.path.isdir(data_dir):
        for f in os.listdir(data_dir):
            path = os.path.join(data_dir, f)
            data_files[f] = {
                "size_mb": round(os.path.getsize(path) / 1_048_576, 2),
            }

    return {
        "available_models": _check_available_models(model_dir),
        "loaded_model": _state["model_type"],
        "model_dir": model_dir,
        "data_files": data_files,
    }
