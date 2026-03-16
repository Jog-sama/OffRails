"""
Pydantic schemas for the Agent Trace Anomaly Detection API.

Matches the interface defined in scripts/inference.py:
    TraceAnomalyDetector.predict() returns:
        is_anomalous: bool
        confidence:   float
        label:        int (0=normal, 1=anomalous)
        anomaly_signals: list[str]
        features:     dict (xgboost only)
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request Schemas ──────────────────────────────────────────────────────────

class TraceMessage(BaseModel):
    """
    A single message in an agent execution trace.
    Accepts both ToolBench/ShareGPT format ('from'/'value')
    and OpenAI format ('role'/'content').
    """
    role: Optional[str] = Field(None, alias="from", description="Message role (OpenAI format)")
    value: Optional[str] = Field(None, description="Message content (ShareGPT format)")
    content: Optional[str] = Field(None, description="Message content (OpenAI format)")

    model_config = {"populate_by_name": True}

    def to_dict(self) -> dict:
        """Normalize to the format inference.py expects."""
        d = {}
        if self.role is not None:
            d["from"] = self.role
        if self.value is not None:
            d["value"] = self.value
        if self.content is not None:
            d["value"] = self.content  # map content → value for ToolBench compat
            if "from" not in d and self.role:
                d["from"] = self.role
        return d


class PredictRequest(BaseModel):
    """Request body for single-trace anomaly prediction."""
    conversations: list[dict] = Field(
        ...,
        description=(
            "List of message dicts in ShareGPT/ToolBench format. "
            "Each dict should have 'from' (role) and 'value' (content) keys."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversations": [
                        {"from": "user", "value": "Find me flights from NYC to London"},
                        {"from": "assistant", "value": "I'll search for flights using the travel API."},
                        {"from": "function", "value": '{"flights": [{"price": 450}]}'},
                        {"from": "assistant", "value": "I found flights starting at $450."},
                    ]
                }
            ]
        }
    }


class PredictBatchRequest(BaseModel):
    """Request body for batch prediction on multiple traces."""
    traces: list[list[dict]] = Field(
        ..., description="List of traces, each trace is a list of message dicts"
    )


class ModelLoadRequest(BaseModel):
    """Request to load a specific model type."""
    model_type: str = Field(
        "xgboost",
        description="Model to load: 'xgboost' or 'distilbert'",
        pattern="^(xgboost|distilbert)$",
    )
    model_dir: str = Field("models", description="Path to saved models directory")


# ── Response Schemas ─────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """Response from the anomaly detector — mirrors TraceAnomalyDetector.predict() output."""
    is_anomalous: bool = Field(..., description="True if the trace is predicted anomalous")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of anomaly")
    label: int = Field(..., description="0 = normal, 1 = anomalous")
    anomaly_signals: list[str] = Field(
        default_factory=list,
        description="Human-readable explanations of detected anomaly patterns",
    )
    model_used: str = Field(..., description="Which model produced this prediction")
    features: Optional[dict] = Field(
        None, description="Extracted feature values (xgboost only)"
    )


class PredictBatchResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[PredictResponse]
    anomaly_count: int
    total: int


class CompareResponse(BaseModel):
    """Side-by-side prediction from both models on the same trace."""
    xgboost: Optional[PredictResponse] = None
    distilbert: Optional[PredictResponse] = None
    agreement: bool = Field(..., description="Whether both models agree on the label")


class HealthResponse(BaseModel):
    status: str
    loaded_model: Optional[str] = None
    available_models: list[str]
    model_dir: str
