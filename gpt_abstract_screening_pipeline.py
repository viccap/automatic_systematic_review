"""
LLM-assisted abstract screening pipeline with prompt revision support.

Key improvements over the prior script:
- No side effects on import (execution is gated by __main__).
- Model selection is respected; no hard-coded model overrides.
- Robust prompt path/version handling and placeholder safety.
- Clean train/val/test split to reduce overfitting risk.
- Per-iteration prediction buffers (no global, cumulative state leakage).
- Safer output parsing with JSON-first fallback to regex.
- Graceful metric handling for degenerate class distributions.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import sklearn.model_selection
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

import critic_prompt
import revision_prompt

MAIN_MODEL = "deepseek-r1"#"qwen3-235b-a22b"
CRITIC_MODEL = "deepseek-r1"
REVISION_MODEL = "deepseek-r1"#"mistral-large-instruct"

PROMPT_VERSION_PATTERN = re.compile(r"(system|user)_prompt_v(?P<version>\d+)\.txt$")
DEFAULT_SYSTEM_PROMPT = Path("prompt_log/system_prompt_v1.txt")
DEFAULT_USER_PROMPT = Path("prompt_log/user_prompt_v1.txt")
DEFAULT_DATASET = Path("V1_abstract_screening.xlsx")

# Keep the model output tightly structured so parsing is reliable.
OUTPUT_FORMAT_HINT = """
---
Output format (required):
Return JSON only, no prose, no markdown fences. Example:
{"decision": "INCLUDE", "rationale": "<one-sentence justification>"}
Allowed decision values: INCLUDE or EXCLUDE.
"""

load_dotenv()


@dataclass
class PromptPaths:
    system_path: Path
    user_path: Path

    def next_version(self, system_text: str, user_text: str) -> "PromptPaths":
        """
        Save the next version of the prompts and return updated paths.
        """
        next_sys = _bump_prompt_path(self.system_path)
        next_user = _bump_prompt_path(self.user_path)
        next_sys.write_text(system_text)
        next_user.write_text(_ensure_placeholders(user_text))
        return PromptPaths(system_path=next_sys, user_path=next_user)



def init_client() -> OpenAI:
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL", "https://chat-ai.academiccloud.de/v1")
    if not api_key:
        raise RuntimeError("API_KEY missing from environment. Set it before running.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _ensure_placeholders(prompt_text: str) -> str:
    parts: List[str] = []
    if "{ABSTRACT}" not in prompt_text:
        parts.append("\n\n---\nABSTRACT:\n{ABSTRACT}")
    if "{TITLE}" not in prompt_text:
        parts.append("\n\n---\nTITLE:\n{TITLE}")
    return prompt_text + "".join(parts)


def _read_prompt(path: Path) -> str:
    return path.read_text()


def _format_user_prompt(template_path: Path, abstract: str, title: str) -> str:
    base = _read_prompt(template_path).format(ABSTRACT=abstract, TITLE=title)
    return base + OUTPUT_FORMAT_HINT


def _bump_prompt_path(path: Path) -> Path:
    """
    Increment a prompt filename version (e.g., system_prompt_v1.txt -> system_prompt_v2.txt).
    """
    match = PROMPT_VERSION_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Prompt filename does not match expected pattern: {path}")
    version = int(match.group("version"))
    next_version = version + 1
    return path.with_name(f"{match.group(1)}_prompt_v{next_version}.txt")


def _extract_vote(output_text: str) -> Optional[str]:
    """
    Extract INCLUDE/EXCLUDE from model output.
    Prefers JSON payloads like {"decision": "INCLUDE"}; falls back to regex.
    """
    if not isinstance(output_text, str):
        return None
    cleaned = output_text.strip()
    # Remove code fences if present.
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    # Try JSON first.
    try:
        parsed = json.loads(cleaned)
        decision = parsed.get("decision") or parsed.get("Decision")
        if isinstance(decision, str):
            decision_upper = decision.strip().upper()
            if decision_upper in {"INCLUDE", "EXCLUDE"}:
                return decision_upper
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback regex.
    match = re.search(r"Decision:\s*(INCLUDE|EXCLUDE)", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Generic containment fallback.
    generic_match = re.search(r"\b(INCLUDE|EXCLUDE)\b", cleaned, re.IGNORECASE)
    if generic_match:
        return generic_match.group(1).upper()
    return None


def _build_comparison_df(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        truth[["short_id", "included"]], predictions[["short_id", "output"]], on="short_id", how="inner"
    ).copy()
    merged["vote"] = merged["output"].apply(_extract_vote)
    if merged["vote"].isnull().any():
        raise ValueError("One or more model outputs lacked a valid INCLUDE/EXCLUDE decision.")
    merged["y_true"] = merged["included"].str.lower().map({"yes": 1, "no": 0})
    merged["y_pred"] = merged["vote"].map({"INCLUDE": 1, "EXCLUDE": 0})
    return merged


def evaluate_predictions(predictions: pd.DataFrame, truth: pd.DataFrame, show_plot: bool = False) -> Tuple[Dict, pd.DataFrame]:
    """
    Compute metrics and return (metrics_dict, false_classifications_df).
    """
    df = _build_comparison_df(predictions, truth)
    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(df["y_true"], df["y_pred"]),
        "precision": precision_score(df["y_true"], df["y_pred"], zero_division=0),
        "recall": recall_score(df["y_true"], df["y_pred"], zero_division=0),
        "confusion_matrix": cm,
    }
    # Warn if we only saw one class; metrics will be unstable.
    if df["y_true"].nunique() < 2 or df["y_pred"].nunique() < 2:
        print("Warning: only one class present in y_true or y_pred; metrics may be uninformative.")
    if show_plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["EXCLUDE", "INCLUDE"])
        disp.plot()
    false_clf = df[df["y_true"] != df["y_pred"]]
    return metrics, false_clf


def split_dataset(df: pd.DataFrame, train_size: float = 0.6, val_size: float = 0.2, seed: int = 42):
    """
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X = df.drop(columns=["included", "article_type"])
    y = df[["short_id", "included"]]
    X_train, X_tmp, y_train, y_tmp = sklearn.model_selection.train_test_split(X, y, test_size=1 - train_size, random_state=seed)
    relative_val = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(
        X_tmp, y_tmp, test_size=1 - relative_val, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_relevance_score(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    short_id: str,
    sleep_seconds: float = 2.0,
) -> Dict:
    """
    Call the screening agent and return a record dict.
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
    )
    output = chat_completion.choices[0].message.content.strip()
    time.sleep(sleep_seconds)  # basic rate limiting
    print(f"Screening output for {short_id}:\n{output}\n")
    return {"short_id": short_id, "output": output}


def critic_agent(
    client: OpenAI,
    row: pd.Series,
    prompt_paths: PromptPaths,
    model: str = CRITIC_MODEL,
) -> str:
    ground_truth_df = pd.read_excel(DEFAULT_DATASET)
    short_id = row["short_id"]
    paper = ground_truth_df.loc[ground_truth_df.short_id == short_id].iloc[0]
    critic_user_prompt = critic_prompt.CRITIC_USER.format(
        TITLE=paper.title,
        ABSTRACT=paper.abstract,
        CHAIN_OF_THOUGHT=row["output"],
        VOTE=row["vote"],
        DECISION=row["included"],
        SYSTEM_PROMPT=_read_prompt(prompt_paths.system_path),
        USER_PROMPT=_read_prompt(prompt_paths.user_path),
    )
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": critic_prompt.CRITIC_SYSTEM}, {"role": "user", "content": critic_user_prompt}],
        model=model,
    )
    print(f"Critic output:\n{chat_completion.choices[0].message.content.strip()}\n")
    return chat_completion.choices[0].message.content.strip()


def revision_agent(
    client: OpenAI,
    row: pd.Series,
    feedback: str,
    prompt_paths: PromptPaths,
    model: str = REVISION_MODEL,
) -> str:
    ground_truth_df = pd.read_excel(DEFAULT_DATASET)
    short_id = row["short_id"]
    paper = ground_truth_df.loc[ground_truth_df.short_id == short_id].iloc[0]
    user_prompt = revision_prompt.REVISION_USER_PROMPT.format(
        title=paper.title,
        abstract=paper.abstract,
        agent_reasoning=row["output"],
        wrong_decision=row["vote"],
        correct_label=row["included"],
        critic_feedback=feedback,
        original_system_prompt=_read_prompt(prompt_paths.system_path),
        original_user_prompt=_read_prompt(prompt_paths.user_path),
    )
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": revision_prompt.REVISION_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        model=model,
    )
    return chat_completion.choices[0].message.content.strip()

def parse_revised_prompts(revised_prompts_text: str, current_prompts: PromptPaths) -> Tuple[str, str]:
    """
    Split revision output into system/user prompts and ensure they differ from the originals.
    """
    if "---REVISED USER PROMPT---" not in revised_prompts_text:
        raise ValueError("Revision output missing delimiter '---REVISED USER PROMPT---'.")
    system_text, user_text = revised_prompts_text.split("---REVISED USER PROMPT---", maxsplit=1)
    system_text = system_text.replace("---REVISED SYSTEM PROMPT---", "", 1).strip()
    user_text = user_text.strip()

    # Strip any inadvertent <think>...</think> reasoning blocks.
    system_text = re.sub(r"<think>.*?</think>", "", system_text, flags=re.DOTALL | re.IGNORECASE).strip()
    user_text = re.sub(r"<think>.*?</think>", "", user_text, flags=re.DOTALL | re.IGNORECASE).strip()

    original_system = _read_prompt(current_prompts.system_path).strip()
    original_user = _read_prompt(current_prompts.user_path).strip()

    if system_text == original_system and user_text == original_user:
        raise ValueError("Revision agent returned prompts identical to the originals; refusing to save.")

    return system_text, user_text


def run_screening_batch(
    client: OpenAI,
    model: str,
    data: pd.DataFrame,
    prompt_paths: PromptPaths,
    limit: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Screen a batch of abstracts and return predictions DataFrame.
    Rows are randomly sampled (deterministically via random_state) to avoid class skew from ordering.
    """
    records: List[Dict] = []
    system_prompt = _read_prompt(prompt_paths.system_path)

    if limit is None:
        batch = data
    else:
        batch = data.sample(n=min(limit, len(data)), random_state=random_state)

    for _, row in batch.iterrows():
        user_prompt = _format_user_prompt(prompt_paths.user_path, abstract=row["abstract"], title=row["title"])
        record = get_relevance_score(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            short_id=row["short_id"],
        )
        records.append(record)
    return pd.DataFrame(records)


def iterative_prompt_optimization(
    ground_truth_path: Path = DEFAULT_DATASET,
    prompt_paths: PromptPaths = PromptPaths(DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT),
    main_model: str = MAIN_MODEL,
    critic_model: str = CRITIC_MODEL,
    revision_model: str = REVISION_MODEL,
    max_iterations: int = 3,
    train_limit: int = 50,
    calls_per_iter: int = 6, # this is fairly large; adjust based on budget
) -> None:
    """
    Run iterative prompt refinement using train for tuning and val for selection.
    """
    df = pd.read_excel(ground_truth_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    client = init_client()

    current_prompts = prompt_paths

    for iteration in range(1, max_iterations + 1):
        print(f"\n=== Iteration {iteration} ===")
        # Train batch
        train_preds = run_screening_batch(
            client=client,
            model=main_model,
            data=X_train.head(train_limit),
            prompt_paths=current_prompts,
            limit=calls_per_iter,
        )
        train_metrics, false_clf = evaluate_predictions(train_preds, y_train, show_plot=False)
        print(f"Train metrics: {train_metrics}")

        if false_clf.empty:
            print("No false classifications; stopping early.")
            break

        # Use the first false classification to revise prompts.
        row = false_clf.iloc[0]
        critic_feedback = critic_agent(client, row, current_prompts, model=critic_model)
        revised_prompts_text = revision_agent(client, row, feedback=critic_feedback, prompt_paths=current_prompts, model=revision_model)

        system_text, user_text = parse_revised_prompts(revised_prompts_text, current_prompts)
        current_prompts = current_prompts.next_version(system_text, user_text)
        print(f"Saved revised prompts: {current_prompts.system_path}, {current_prompts.user_path}")

        # Evaluate on validation to check for improvement (optional thresholding could be added).
        val_preds = run_screening_batch(
            client=client,
            model=main_model,
            data=X_val,
            prompt_paths=current_prompts,
            limit=calls_per_iter,
        )
        val_metrics, _ = evaluate_predictions(val_preds, y_val, show_plot=False)
        print(f"Validation metrics: {val_metrics}")

    # Final test evaluation with the latest prompts.
    test_preds = run_screening_batch(
        client=client,
        model=main_model,
        data=X_test,
        prompt_paths=current_prompts,
        limit=calls_per_iter,
    )
    test_metrics, _ = evaluate_predictions(test_preds, y_test, show_plot=False)
    print(f"\nFinal test metrics (latest prompts): {test_metrics}")


if __name__ == "__main__":
    iterative_prompt_optimization()
