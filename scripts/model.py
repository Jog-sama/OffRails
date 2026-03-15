"""
Defines all three required modeling approaches for agent trace anomaly detection:

1. NaiveBaseline          : majority-class classifier (always predicts the most common label)
2. ClassicalMLModel       : XGBoost gradient boosting on handcrafted trace features
3. TraceTransformer       : DistilBERT fine-tuned on raw trace text as a sequence classifier

Each model exposes a consistent interface:
    .fit(X, y)  / .train(...)
    .predict(X) : np.ndarray of labels
    .predict_proba(X) : np.ndarray of probabilities
    .save(path)
    .load(path)  (classmethod)
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from xgboost import XGBClassifier


# Naive Baseline

class NaiveBaseline:
    """
    Majority-class classifier. Predicts the most frequent label in training data.
    This serves as the lower-bound baseline.
    """

    def __init__(self):
        self.majority_class = 0
        self.class_prior = 0.5

    def fit(self, X, y):
        """Learn the majority class from training labels."""
        y = np.array(y)
        counts = np.bincount(y)
        self.majority_class = int(np.argmax(counts))
        self.class_prior = counts[self.majority_class] / len(y)
        print(f"[NaiveBaseline] Majority class = {self.majority_class} "
              f"(prior = {self.class_prior:.3f})")
        return self

    def predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.full(n, self.majority_class)

    def predict_proba(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        proba = np.zeros((n, 2))
        proba[:, self.majority_class] = self.class_prior
        proba[:, 1 - self.majority_class] = 1 - self.class_prior
        return proba

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"majority_class": self.majority_class, "class_prior": self.class_prior}, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        model = cls()
        model.majority_class = data["majority_class"]
        model.class_prior = data["class_prior"]
        return model


# Classical ML Model (XGBoost)

class ClassicalMLModel:
    """
    XGBoost classifier trained on handcrafted trace features.
    Handles class imbalance via scale_pos_weight.
    """

    def __init__(self, params: dict = None):
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None
        self.feature_names = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train XGBoost with optional early stopping on validation set.
        Automatically computes scale_pos_weight for class imbalance.
        """
        y = np.array(y)
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        scale = n_neg / n_pos if n_pos > 0 else 1.0
        self.params["scale_pos_weight"] = scale

        print(f"[XGBoost] Training with scale_pos_weight = {scale:.2f}")
        print(f"[XGBoost] Class distribution: 0={n_neg}, 1={n_pos}")

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        self.model = XGBClassifier(**self.params)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]

        self.model.fit(X, y, verbose=50, **fit_params)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances as a sorted Series."""
        names = self.feature_names or [f"f{i}" for i in range(len(self.model.feature_importances_))]
        return pd.Series(
            self.model.feature_importances_, index=names
        ).sort_values(ascending=False)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        return instance


# Deep Learning Model (Transformer) - DistilBERT fine-tuned on raw trace text

class TraceDataset(Dataset):
    """PyTorch dataset for tokenized agent traces."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TraceTransformer:
    """
    DistilBERT fine-tuned for binary classification on raw agent trace text.
    Truncates traces to max_length tokens — captures the most salient parts
    of the execution trace for anomaly signals.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        device: str = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None

    def _init_model(self):
        """Lazy-init tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        ).to(self.device)

    def fit(self, X_train: list[str], y_train: list[int],
            X_val: list[str] = None, y_val: list[int] = None):
        """
        Fine-tune DistilBERT on trace text.
        Handles class imbalance via weighted cross-entropy.
        """
        self._init_model()

        # compute class weights
        y_arr = np.array(y_train)
        n_neg = np.sum(y_arr == 0)
        n_pos = np.sum(y_arr == 1)
        weight_pos = n_neg / n_pos if n_pos > 0 else 1.0
        class_weights = torch.tensor([1.0, weight_pos], dtype=torch.float).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        print(f"[Transformer] Training on {len(X_train)} samples, device={self.device}")
        print(f"[Transformer] Class weights: [1.0, {weight_pos:.2f}]")

        train_ds = TraceDataset(X_train, y_train, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val and y_val:
            val_ds = TraceDataset(X_val, y_val, self.tokenizer, self.max_length)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_val_f1 = 0.0
        best_state = None

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            n_correct = 0
            n_total = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
                loss = loss_fn(outputs.logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                n_correct += (preds == labels).sum().item()
                n_total += len(labels)

            train_acc = n_correct / n_total
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch + 1}/{self.num_epochs} — "
                  f"loss: {avg_loss:.4f}, train_acc: {train_acc:.4f}", end="")

            # validation
            if val_loader:
                val_preds, val_labels = self._predict_loader(val_loader)
                from sklearn.metrics import f1_score
                val_f1 = f1_score(val_labels, val_preds, average="macro")
                print(f", val_f1: {val_f1:.4f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                print()

        # restore best model
        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            print(f"[Transformer] Restored best model (val_f1 = {best_val_f1:.4f})")

        return self

    def _predict_loader(self, loader):
        """Run inference on a dataloader, return predictions and labels."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"]

                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        return np.array(all_preds), np.array(all_labels)

    def predict(self, X: list[str]) -> np.ndarray:
        """Predict labels for a list of raw trace texts."""
        ds = TraceDataset(X, [0] * len(X), self.tokenizer, self.max_length)
        loader = DataLoader(ds, batch_size=self.batch_size)
        preds, _ = self._predict_loader(loader)
        return preds

    def predict_proba(self, X: list[str]) -> np.ndarray:
        """Return class probabilities for a list of raw trace texts."""
        self.model.eval()
        ds = TraceDataset(X, [0] * len(X), self.tokenizer, self.max_length)
        loader = DataLoader(ds, batch_size=self.batch_size)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def save(self, path: str):
        """Save model and tokenizer to directory."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # save config
        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        }
        with open(os.path.join(path, "trace_config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str, device: str = None):
        """Load a saved model from directory."""
        with open(os.path.join(path, "trace_config.json")) as f:
            config = json.load(f)

        instance = cls(
            model_name=path,  # load from local dir
            max_length=config["max_length"],
            batch_size=config["batch_size"],
            device=device,
        )
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path).to(instance.device)
        return instance
