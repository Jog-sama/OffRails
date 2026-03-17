# Agent Trace Anomaly Detection

Detect when AI agents "go off the rails" — unnecessary tool calls, circular reasoning, and goal drift — by framing multi-step execution traces as a **sequence anomaly detection** problem.

## Problem Statement

LLM-based agents (e.g., ReAct, ToolLLM) execute multi-step tool-calling workflows. Sometimes these traces exhibit failure modes:
- **Circular reasoning**: calling the same tool repeatedly with no progress
- **Goal drift**: diverging from the original user intent
- **Unnecessary tool calls**: invoking irrelevant APIs
- **Silent failures**: completing without actually answering the query

We build a binary classifier that takes an agent execution trace (sequence of tool calls, reasoning steps, observations) and predicts whether the trace is **anomalous** (1) or **normal** (0).

## Dataset

**Source**: [ToolBench v1](https://huggingface.co/datasets/tuandunghcmut/toolbench-v1) (Qin et al., ICLR 2024)

**Proxy labeling**: Since ToolBench doesn't include explicit pass/fail labels, we construct proxy anomaly labels by analyzing the final assistant message:
- Contains failure language ("I cannot", "failed", "unable to") → **anomalous**
- Zero tool calls (Action: ...) in the trace → **anomalous**
- Otherwise → **normal**

This labeling is intentionally imperfect — Experiment 2 (noise robustness) directly quantifies how sensitive our models are to these proxy label errors.

## Models

| Model | Type | Input | Description |
|-------|------|-------|-------------|
| **Naive Baseline** | Majority class | Labels only | Always predicts the most frequent class |
| **XGBoost** | Classical ML | 25+ handcrafted features | Gradient boosting on structural, behavioral, and linguistic features extracted from traces |
| **DistilBERT** | Deep Learning | Compact trace text | Fine-tuned transformer on a focused input: user query + last reasoning step + last tool response + final answer |

### Handcrafted Features (XGBoost)

- **Structural**: turn count, trace length, conversation depth
- **Tool-usage**: call count, diversity ratio, tool call density
- **Behavioral**: consecutive same-tool calls, repeated calls, call-response ratio
- **Linguistic**: error keywords, apology phrases, hedging language, give-up signals
- **Positional**: where tool calls appear in the trace (early vs. late)
- **Observation quality**: error observations, empty responses

### Compact Trace Input (DistilBERT)

Raw agent traces are 2000–5000+ tokens, far exceeding DistilBERT's 512-token context window. Naive truncation cuts off the end of the trace — exactly where the anomaly signals live (failure language, give-up patterns in the final response). Our solution constructs a compact ~300 token input per trace:

```
[QUERY] The original user request
[REASONING] The second-to-last assistant turn (shows reasoning trajectory)
[LAST_OBS] The final tool response (shows if API calls succeeded)
[RESPONSE] The last assistant turn (where failure/success is most evident)
```

This ensures the model sees all critical signal with zero information loss from truncation. Additionally, the tokenizer uses **tail truncation** (keeps the last 512 tokens rather than the first) as a safety net for longer traces.

## Trained Models

**Note**: Trained model files (`models/` directory) are not included in this repository as they exceed GitHub's file size limits. To reproduce:

1. Run the training pipeline (see Quick Start below), **or**
2. Use the provided [Colab notebook](notebooks/OffRails_Train.ipynb) to train on GPU and download the model artifacts

The expected contents of `models/` after training:
```
models/
├── naive_baseline.joblib          (~1 KB)
├── xgboost_model.joblib           (~2 MB)
└── distilbert_trace/              (~260 MB)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.txt
    └── trace_config.json
```

## Project Structure

```
├── README.md               ← this file
├── requirements.txt        ← Python dependencies
├── setup.py                ← orchestrates the full pipeline
├── main.py                 ← main entry point (pipeline / inference / demo)
├── scripts/
│   ├── make_dataset.py     ← data download, preprocessing, proxy labeling
│   ├── build_features.py   ← handcrafted feature extraction
│   ├── model.py            ← all three model definitions
│   ├── train.py            ← training orchestration
│   ├── evaluate.py         ← metrics, confusion matrices, error analysis
│   ├── experiment.py       ← sensitivity + noise robustness experiments
│   ├── tune_hyperparams.py ← XGBoost hyperparameter grid search
│   └── inference.py        ← production inference module (for FastAPI)
├── models/                 ← saved trained models (not in repo — see note above)
├── data/
│   ├── raw/                ← raw downloaded data
│   ├── processed/          ← train/val/test splits + features
│   └── outputs/            ← plots, metrics, experiment results
├── notebooks/              ← Colab training notebook + exploration
└── .gitignore
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python setup.py
```

This will:
1. Download ToolBench data and create proxy labels
2. Extract handcrafted features
3. Train all three models (naive, XGBoost, DistilBERT)
4. Evaluate on test set with full metrics
5. Run experiments (sensitivity + noise robustness)

### 3. Quick Test (subset of data)
```bash
python setup.py --max_samples 5000
```

### 4. Train Individual Models
```bash
python setup.py --step data
python setup.py --step features
python setup.py --step train --model classical   # XGBoost only
python setup.py --step train --model deep         # DistilBERT only
python setup.py --step evaluate
```

### 5. Train on Google Colab (recommended for DistilBERT)
Upload `notebooks/OffRails_Train.ipynb` to Google Colab, select a GPU runtime (L4/T4/A100), and run all cells. Download the resulting `trained_models.zip` and unzip into `models/`.

### 6. Run Inference
```bash
python main.py inference --trace path/to/trace.json --model_type xgboost
```

### 7. Interactive Demo
```bash
python main.py demo
```

## Experiments

### Experiment 1: Training Set Size Sensitivity
Trains XGBoost at 10%, 25%, 50%, 75%, 100% of data with 3 random seeds each.
**Motivation**: Determines if we need more data or better features.

### Experiment 2: Label Noise Robustness
Flips 0–25% of training labels randomly to simulate proxy label errors.
**Motivation**: Our labels are heuristic-based, so quantifying noise sensitivity is directly relevant. If the model is robust to 15%+ noise, our proxy labeling strategy is viable.

## Integration with Backend

The `scripts/inference.py` module provides a `TraceAnomalyDetector` class for the backend import:

```python
from scripts.inference import TraceAnomalyDetector

detector = TraceAnomalyDetector(model_dir="models", model_type="xgboost")
result = detector.predict(conversation_json)
# result = {
#   "is_anomalous": True/False,
#   "confidence": 0.87,
#   "label": 0 or 1,
#   "anomaly_signals": ["Circular behavior detected: ...", ...]
# }
```

## Evaluation Metrics

- **Primary**: F1 Score (binary, on anomalous class) — balances precision and recall for the minority class
- **Secondary**: Macro F1, ROC AUC, Precision-Recall curves
- **Justification**: Standard accuracy is misleading with class imbalance. F1 directly measures our ability to catch anomalous traces while avoiding false alarms.

## References

- Qin et al., "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs", ICLR 2024
- [ToolBench GitHub](https://github.com/OpenBMB/ToolBench)
- [ToolBench HuggingFace Dataset](https://huggingface.co/datasets/tuandunghcmut/toolbench-v1)

## Team

- **Atharva** — ML pipeline, model training, evaluation, experiments
- **Omkar** — Backend (FastAPI)
- **Mrinal** — Frontend

## AI Attribution

Parts of this codebase were developed with the assistance of Claude (Anthropic). All AI-generated code has been reviewed, tested, and adapted by the team.