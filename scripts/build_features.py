"""
This script extracts handcrafted features from agent execution traces for the classical
ML pipeline. These features capture structural, behavioral, and linguistic
signals of trace anomalies (circular reasoning, goal drift, unnecessary
tool calls, failure patterns).

Feature groups:
  1. Structural     : turn counts, trace length, conversation depth
  2. Tool-usage     : call counts, diversity, repetition, density
  3. Behavioral     : circular patterns, consecutive repeats, call-response ratio
  4. Linguistic     : error keywords, sentiment signals, response length stats
  5. Sequence       : positional features, where tool calls appear in the trace
"""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


# list of keywords/phrases indicative of errors, failures, or issues in tool responses
ERROR_KEYWORDS = [
    "error", "failed", "failure", "exception", "timeout",
    "unauthorized", "forbidden", "not found", "invalid",
    "rate limit", "500", "502", "503", "404", "400",
]

APOLOGY_KEYWORDS = [
    "sorry", "apologize", "unfortunately", "regret",
    "unable", "cannot", "can't", "couldn't",
]

HEDGING_KEYWORDS = [
    "might", "perhaps", "possibly", "try again",
    "not sure", "uncertain", "unclear",
]

GIVE_UP_KEYWORDS = [
    "give up", "stop here", "will not", "won't be able",
    "no further", "end here", "conclude",
]

# regex to detect Action/Action Input inside gpt turns (ReAct format)
ACTION_RE = re.compile(r"Action\s*:\s*(.+)", re.IGNORECASE)


def normalize_message(msg):
    """
    Ensure a conversation message is a dict with 'from' and 'value' keys.
    ToolBench stores some turns as serialized JSON strings instead of dicts.
    """
    if isinstance(msg, str):
        try:
            msg = json.loads(msg)
        except (json.JSONDecodeError, TypeError):
            msg = {"from": "unknown", "value": msg}

    if not isinstance(msg, dict):
        msg = {"from": "unknown", "value": str(msg)}

    return msg


def extract_actions_from_gpt_turn(text: str) -> list[str]:
    """
    Extract tool call names from a gpt/assistant turn that uses ReAct format.
    Returns list of action names found in the text.
    """
    actions = ACTION_RE.findall(text)
    cleaned = []
    for a in actions:
        a = a.strip()
        if a.lower() not in ("finish", "none", ""):
            cleaned.append(a)
    return cleaned


def count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count how many keyword patterns appear in text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def extract_tool_names(tool_calls: list[str]) -> list[str]:
    """
    Extract function/API names from tool call strings.
    ToolBench tool calls are typically JSON with a 'name' or 'api_name' field,
    OR plain action names extracted from ReAct-format gpt turns.
    """
    names = []
    for tc in tool_calls:
        # first try parsing as JSON (explicit function_call format)
        try:
            parsed = json.loads(tc)
            name = parsed.get("name", parsed.get("api_name", parsed.get("function", "")))
            if name:
                names.append(name)
                continue
            else:
                for key in parsed:
                    if "name" in key.lower():
                        names.append(str(parsed[key]))
                        break
                continue
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        # fallback: regex for common patterns
        match = re.search(r'"(?:name|api_name|function)"\s*:\s*"([^"]+)"', tc)
        if match:
            names.append(match.group(1))
        else:
            # already a plain action name from ReAct extraction
            names.append(tc.strip()[:80])
    return names


def compute_repetition_features(tool_names: list[str]) -> dict:
    """
    Compute features related to repeated/circular tool usage.
    """
    if not tool_names:
        return {
            "unique_tools": 0,
            "tool_diversity_ratio": 0.0,
            "max_single_tool_freq": 0,
            "max_consecutive_same_tool": 0,
            "num_repeated_exact_calls": 0,
        }

    counts = Counter(tool_names)
    unique = len(counts)
    total = len(tool_names)

    # longest streak of the same tool called consecutively
    max_consec = 1
    current_consec = 1
    for i in range(1, len(tool_names)):
        if tool_names[i] == tool_names[i - 1]:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 1

    return {
        "unique_tools": unique,
        "tool_diversity_ratio": unique / total if total > 0 else 0.0,
        "max_single_tool_freq": max(counts.values()),
        "max_consecutive_same_tool": max_consec,
        "num_repeated_exact_calls": total - unique,
    }


def compute_response_length_features(assistant_turns: list[str]) -> dict:
    """Stats on assistant response lengths."""
    if not assistant_turns:
        return {
            "mean_response_len": 0.0,
            "std_response_len": 0.0,
            "max_response_len": 0,
            "min_response_len": 0,
            "total_response_chars": 0,
        }

    lengths = [len(t) for t in assistant_turns]
    return {
        "mean_response_len": float(np.mean(lengths)),
        "std_response_len": float(np.std(lengths)),
        "max_response_len": max(lengths),
        "min_response_len": min(lengths),
        "total_response_chars": sum(lengths),
    }


def compute_observation_features(observations: list[str]) -> dict:
    """Features from tool responses/observations."""
    if not observations:
        return {
            "mean_obs_len": 0.0,
            "num_error_observations": 0,
            "num_empty_observations": 0,
        }

    lengths = [len(o) for o in observations]
    n_errors = sum(
        1 for o in observations
        if count_keyword_hits(o, ERROR_KEYWORDS) > 0
    )
    n_empty = sum(1 for o in observations if len(o.strip()) < 10)

    return {
        "mean_obs_len": float(np.mean(lengths)),
        "num_error_observations": n_errors,
        "num_empty_observations": n_empty,
    }


def compute_positional_features(turns: list[tuple], tool_call_turn_indices: list[int]) -> dict:
    """
    Where in the trace do tool calls appear?
    Normalized position (0 = start, 1 = end).
    """
    total = len(turns) if turns else 1

    if not tool_call_turn_indices:
        return {
            "first_tool_position": 1.0,
            "last_tool_position": 0.0,
            "tool_position_spread": 0.0,
        }

    positions = [i / total for i in tool_call_turn_indices]

    return {
        "first_tool_position": positions[0],
        "last_tool_position": positions[-1],
        "tool_position_spread": positions[-1] - positions[0],
    }


def extract_features_from_row(row: pd.Series) -> dict:
    """
    Extract all features from a single processed dataset row.
    """
    conv = json.loads(row["conversations_json"])

    # rebuild parsed trace
    turns = []
    tool_calls = []
    observations = []
    assistant_turns = []
    tool_call_turn_indices = []  # which turn indices contain tool calls

    for raw_msg in conv:
        msg = normalize_message(raw_msg)
        role = msg.get("from", msg.get("role", "unknown"))
        content = msg.get("value", msg.get("content", ""))
        if content is None:
            content = ""

        turn_idx = len(turns)
        turns.append((role, content))

        if role in ("gpt", "assistant", "chatgpt"):
            assistant_turns.append(content)
            # extract embedded tool calls from ReAct-style gpt turns
            actions = extract_actions_from_gpt_turn(content)
            if actions:
                tool_calls.extend(actions)
                tool_call_turn_indices.append(turn_idx)
        elif role in ("function_call", "tool_call"):
            tool_calls.append(content)
            tool_call_turn_indices.append(turn_idx)
        elif role in ("observation", "tool_response", "function"):
            observations.append(content)

    # extract tool names for repetition analysis
    tool_names = extract_tool_names(tool_calls)

    # structural features
    features = {
        "num_turns": len(turns),
        "num_tool_calls": len(tool_calls),
        "num_observations": len(observations),
        "num_assistant_turns": len(assistant_turns),
        "trace_length_chars": len(row["raw_trace"]),
    }

    # tool usage features
    features["tool_call_density"] = (
        len(tool_calls) / len(turns) if len(turns) > 0 else 0.0
    )
    features["tool_obs_ratio"] = (
        len(observations) / len(tool_calls) if len(tool_calls) > 0 else 0.0
    )

    # repetition and circularity features
    features.update(compute_repetition_features(tool_names))

    # response length features
    features.update(compute_response_length_features(assistant_turns))

    # observation features
    features.update(compute_observation_features(observations))

    # positional features
    features.update(compute_positional_features(turns, tool_call_turn_indices))

    # linguistic features
    full_text = " ".join(content for _, content in turns)
    last_assistant = assistant_turns[-1] if assistant_turns else ""

    features["error_keyword_count"] = count_keyword_hits(full_text, ERROR_KEYWORDS)
    features["apology_keyword_count"] = count_keyword_hits(full_text, APOLOGY_KEYWORDS)
    features["hedging_keyword_count"] = count_keyword_hits(full_text, HEDGING_KEYWORDS)
    features["give_up_keyword_count"] = count_keyword_hits(full_text, GIVE_UP_KEYWORDS)
    features["last_turn_error_keywords"] = count_keyword_hits(last_assistant, ERROR_KEYWORDS)
    features["last_turn_apology_keywords"] = count_keyword_hits(last_assistant, APOLOGY_KEYWORDS)

    return features


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature extraction to every row. Returns a DataFrame of features + label.
    """
    print(f"[INFO] Extracting features from {len(df)} traces...")
    feature_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        feats = extract_features_from_row(row)
        feats["id"] = row["id"]
        feats["label"] = row["label"]
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows)
    print(f"[INFO] Extracted {len(feat_df.columns) - 2} features.")  # minus id and label
    return feat_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (everything except id and label)."""
    return [c for c in df.columns if c not in ("id", "label")]


def main():
    parser = argparse.ArgumentParser(description="Build features from processed traces")
    parser.add_argument("--input_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        input_path = os.path.join(args.input_dir, f"{split_name}.parquet")
        if not os.path.exists(input_path):
            print(f"[WARN] {input_path} not found, skipping.")
            continue

        df = pd.read_parquet(input_path)
        feat_df = build_feature_matrix(df)

        output_path = os.path.join(args.output_dir, f"{split_name}_features.parquet")
        feat_df.to_parquet(output_path, index=False)
        print(f"[INFO] Saved features → {output_path}")

    print("[DONE] Feature extraction complete.")


if __name__ == "__main__":
    main()