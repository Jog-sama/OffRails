"""
Agent Trace Anomaly Detection — FastAPI Backend

This is the API layer that wraps the ML pipeline built in scripts/.
All model training, feature extraction, and inference logic lives
in the partner's code (scripts/inference.py). This file just serves it.

Run from the OffRails project root:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import os
import sys
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Make partner's scripts/ importable ───────────────────────────────────────
# inference.py does `from model import ...` and `from build_features import ...`
# so we need scripts/ on sys.path.
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from app.api.routes import router

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agent Trace Anomaly Detection API",
    description=(
        "Detects anomalous agent execution traces — unnecessary tool calls, "
        "circular reasoning, and goal drift.\n\n"
        "**ML models** (XGBoost, DistilBERT) are trained via the pipeline in `scripts/`.\n"
        "**This API** serves predictions from those trained models.\n\n"
        "## Workflow\n"
        "1. Train models: `python setup.py` (or `POST /pipeline/train`)\n"
        "2. Load a model: `POST /models/load`\n"
        "3. Predict: `POST /predict`\n"
        "4. Compare models: `POST /predict/compare`\n"
    ),
    version="1.0.0",
)

# Allow Gradio / any frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Agent Trace Anomaly Detection API",
        "docs": "/docs",
        "workflow": [
            "1. Train models: python setup.py",
            "2. POST /models/load  (load xgboost or distilbert)",
            "3. POST /predict      (classify a trace)",
            "4. POST /predict/compare (run both models)",
        ],
    }
