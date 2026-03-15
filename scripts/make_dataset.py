"""
Downloads ToolBench conversation data, constructs proxy anomaly labels,
and saves processed splits to data/processed/.

IMPORTANT: ToolBench stores conversations as a dict of two parallel lists:
    {"from": ["system", "user", "gpt", ...], "value": ["...", "...", "...", ...]}
NOT as a list of dicts. This script handles that format.

Proxy labeling strategy:
  - Look at the LAST assistant message in each conversation.
  - If it contains failure indicators → label = 1 (anomalous).
  - Zero tool calls (Action: ...) in the trace → label = 1 (anomalous).
  - Otherwise → label = 0 (normal).

Source: https://huggingface.co/datasets/tuandunghcmut/toolbench-v1
Paper:  Qin et al., "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs", ICLR 2024.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# ── failure indicators in the last assistant turn ──────────────────────────
FAILURE_PATTERNS = [
    r"i cannot",
    r"i can't",
    r"i'm sorry",
    r"failed",
    r"unable to",
    r"error occurred",
    r"apologize",
    r"unfortunately",
    r"not possible",
    r"no results",
    r"couldn't find",
    r"don't have access",
    r"not available",
    r"give up",
    r"i will stop",
    r"give_up_and_restart",
]

FAILURE_RE = re.compile("|".join(FAILURE_PATTERNS), re.IGNORECASE)

# regex to detect Action: lines inside gpt turns (ReAct format used by ToolBench)
ACTION_RE = re.compile(r"Action\s*:\s*(.+)", re.IGNORECASE)


def normalize_conversations(conv) -> list[dict]:
    """
    Convert ToolBench conversation format into a flat list of message dicts.

    ToolBench stores conversations as:
        {"from": ["system", "user", "gpt", ...], "value": ["...", "...", "...", ...]}

    This function converts to:
        [{"from": "system", "value": "..."}, {"from": "user", "value": "..."}, ...]
    """
    if isinstance(conv, list):
        # already a list — normalize each element
        result = []
        for item in conv:
            if isinstance(item, dict) and "from" in item and "value" in item:
                # already correct format
                if isinstance(item["from"], str):
                    result.append(item)
                else:
                    # nested parallel lists somehow
                    pass
            elif isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        result.append(parsed)
                    else:
                        result.append({"from": "unknown", "value": item})
                except (json.JSONDecodeError, TypeError):
                    result.append({"from": "unknown", "value": item})
            else:
                result.append({"from": "unknown", "value": str(item)})
        return result

    if isinstance(conv, dict):
        # ToolBench parallel-list format: {"from": [...], "value": [...]}
        froms = conv.get("from", [])
        values = conv.get("value", [])

        if isinstance(froms, list) and isinstance(values, list):
            return [
                {"from": str(f), "value": str(v) if v is not None else ""}
                for f, v in zip(froms, values)
            ]

        # single message dict
        if isinstance(froms, str):
            return [{"from": froms, "value": str(values) if values else ""}]

    # fallback
    return []


def extract_actions_from_text(text: str) -> list[str]:
    """
    Extract tool call (Action) names from a gpt turn using ReAct format.
    Ignores 'Finish' action (end-of-task marker).
    """
    actions = ACTION_RE.findall(text)
    cleaned = []
    for a in actions:
        a = a.strip()
        if a.lower() not in ("finish", "none", ""):
            cleaned.append(a)
    return cleaned


def parse_conversation(conv) -> dict:
    """
    Parse a ToolBench conversation into a structured trace dict.
    """
    messages = normalize_conversations(conv)

    turns = []
    tool_calls = []
    observations = []
    assistant_turns = []
    system_prompt = ""
    user_query = ""

    for msg in messages:
        role = msg.get("from", "unknown")
        content = msg.get("value", "")
        if content is None:
            content = ""

        turns.append((role, content))

        if role == "system":
            system_prompt = content
        elif role in ("human", "user") and not user_query:
            user_query = content
        elif role in ("gpt", "assistant", "chatgpt"):
            assistant_turns.append(content)
            # ToolBench embeds tool calls inside gpt turns as "Action: api_name"
            actions = extract_actions_from_text(content)
            tool_calls.extend(actions)
        elif role in ("function_call", "tool_call"):
            tool_calls.append(content)
        elif role in ("observation", "tool_response", "function"):
            observations.append(content)

    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "observations": observations,
        "assistant_turns": assistant_turns,
        "system_prompt": system_prompt,
        "user_query": user_query,
    }


def label_trace(parsed: dict) -> int:
    """
    Assign a proxy anomaly label.
    Returns 1 (anomalous) if failure signals present, 0 otherwise.
    """
    if not parsed["assistant_turns"]:
        return 1

    last_assistant = parsed["assistant_turns"][-1].lower()

    if FAILURE_RE.search(last_assistant):
        return 1

    if len(parsed["tool_calls"]) == 0:
        return 1

    return 0


def extract_raw_trace_text(conv) -> str:
    """Flatten conversation into a single text string for DL models."""
    messages = normalize_conversations(conv)
    parts = []
    for msg in messages:
        role = msg.get("from", "unknown")
        content = msg.get("value", "")
        if content is None:
            content = ""
        parts.append(f"[{role.upper()}] {content}")
    return "\n".join(parts)


def process_dataset(max_samples: int = None) -> pd.DataFrame:
    """Load ToolBench, parse traces, assign labels, return DataFrame."""
    print("[INFO] Loading ToolBench dataset (default config)...")

    try:
        ds = load_dataset("tuandunghcmut/toolbench-v1", "default", split="train")
    except Exception:
        ds = load_dataset("tuandunghcmut/toolbench-v1", split="train")

    print(f"[INFO] Loaded {len(ds)} raw samples.")

    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))
        print(f"[INFO] Subsampled to {max_samples} samples.")

    # debug: peek at first sample structure
    first = ds[0]
    conv_raw = first.get("conversations", {})
    print(f"[DEBUG] conversations type: {type(conv_raw)}")
    if isinstance(conv_raw, dict):
        print(f"[DEBUG] conversations keys: {list(conv_raw.keys())}")
        for k, v in conv_raw.items():
            print(f"[DEBUG]   {k}: type={type(v)}, len={len(v) if isinstance(v, list) else 'N/A'}")
            if isinstance(v, list) and len(v) > 0:
                print(f"[DEBUG]   {k}[0]: {str(v[0])[:120]}")
    elif isinstance(conv_raw, list):
        print(f"[DEBUG] conversations is list, len={len(conv_raw)}")
        if len(conv_raw) > 0:
            print(f"[DEBUG]   [0] type={type(conv_raw[0])}, preview={str(conv_raw[0])[:120]}")

    records = []
    skipped = 0
    for idx, example in enumerate(ds):
        conv = example.get("conversations", {})
        messages = normalize_conversations(conv)

        if not messages:
            skipped += 1
            continue

        try:
            parsed = parse_conversation(conv)
            label = label_trace(parsed)
            raw_text = extract_raw_trace_text(conv)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"[WARN] Skipping sample {idx}: {e}")
            continue

        records.append({
            "id": example.get("id", str(idx)),
            "user_query": parsed["user_query"][:500],
            "num_turns": len(parsed["turns"]),
            "num_tool_calls": len(parsed["tool_calls"]),
            "num_observations": len(parsed["observations"]),
            "num_assistant_turns": len(parsed["assistant_turns"]),
            "raw_trace": raw_text,
            "conversations_json": json.dumps(messages),  # save as list-of-dicts
            "label": label,
        })

    if skipped > 0:
        print(f"[WARN] Skipped {skipped} malformed samples.")

    df = pd.DataFrame(records)
    print(f"[INFO] Processed {len(df)} traces.")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts().to_string()}")
    print(f"[INFO] Anomaly rate: {df['label'].mean():.2%}")
    print(f"[INFO] Tool calls stats:\n{df['num_tool_calls'].describe()}")

    return df


def split_and_save(df, output_dir, test_size=0.15, val_size=0.15, seed=42):
    """Stratified train/val/test split. Saves as parquet."""
    os.makedirs(output_dir, exist_ok=True)

    if df["label"].nunique() < 2:
        print("[WARN] Only one class found. Using random splits.")
        train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
        relative_val = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=relative_val, random_state=seed)
    else:
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["label"]
        )
        relative_val = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=relative_val, random_state=seed, stratify=train_val["label"]
        )

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"{name}.parquet")
        split_df.to_parquet(path, index=False)
        print(f"[INFO] Saved {name}: {len(split_df)} samples → {path}")
        print(f"       Label dist: {dict(split_df['label'].value_counts())}")


def main():
    parser = argparse.ArgumentParser(description="Build ToolBench anomaly detection dataset")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    df = process_dataset(max_samples=args.max_samples)
    split_and_save(df, args.output_dir, args.test_size, args.val_size)
    print("[DONE] Dataset ready.")


if __name__ == "__main__":
    main()