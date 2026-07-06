"""LiveMathematicianBench prompt-assembly, choice-shuffling, and evaluation,
vendored from SkillOpt.

Critical: the choice-shuffling logic (`_shuffle_item_choices`,
`_item_shuffle_seed`) MUST stay byte-faithful with SkillOpt, otherwise the
materialized choice order — and `correct_choice.label` — diverges and our
results stop being comparable to theirs.

Sources:
- _build_system / _build_user / _format_choices: from
  skillopt/envs/livemathematicianbench/rollout.py
- _shuffle_item_choices / _item_shuffle_seed: from
  skillopt/envs/livemathematicianbench/dataloader.py
- normalize_label / extract_answer / parse_choice_label / evaluate: from
  skillopt/envs/livemathematicianbench/evaluator.py
"""
from __future__ import annotations

import hashlib
import random
import re

from .external_prompts import load_prompt


_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


# ── Prompt assembly (vendored from SkillOpt) ─────────────────────────────────


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="livemathc").format(skill_section=skill_section)


def _format_choices(choices: list[dict]) -> str:
    return "\n".join(
        f"{choice['label']}. {choice['text']}"
        for choice in choices
    )


def _build_user(
    item: dict,
    *,
    use_theorem: bool = False,
    use_sketch: bool = False,
) -> str:
    parts = [
        f"## Question\n{item['question']}",
        f"## Choices\n{_format_choices(item['choices'])}",
    ]
    if use_theorem and item.get("theorem"):
        parts.append(f"## Theorem\n{item['theorem']}")
    if use_sketch and item.get("sketch"):
        parts.append(f"## Proof Sketch\n{item['sketch']}")
    return "\n\n".join(parts)


# ── Choice shuffling (vendored from SkillOpt dataloader) ─────────────────────


def _item_shuffle_seed(item_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _shuffle_item_choices(item: dict, seed: int) -> dict:
    """Shuffle choices deterministically per (item_id, seed) and remap correct_choice.label.

    Identical algorithm to SkillOpt's
    LiveMathematicianBenchDataLoader._shuffle_item_choices, so the same
    (item_id, seed) produces the same materialised choice order on both sides.
    """
    shuffled_choices = [dict(c) for c in item["choices"]]
    rng = random.Random(_item_shuffle_seed(str(item["id"]), seed))
    rng.shuffle(shuffled_choices)

    original_correct = normalize_label(item["correct_choice"]["label"])
    remapped_choices: list[dict] = []
    new_correct_choice = dict(item["correct_choice"])

    for idx, choice in enumerate(shuffled_choices):
        new_label = _CHOICE_LABELS[idx]
        old_label = normalize_label(choice["label"])
        remapped_choices.append({"label": new_label, "text": choice["text"]})
        if old_label == original_correct:
            new_correct_choice = {"label": new_label, "text": choice["text"]}

    transformed = dict(item)
    transformed["choices"] = remapped_choices
    transformed["correct_choice"] = new_correct_choice
    return transformed


# ── Evaluation (vendored from SkillOpt) ──────────────────────────────────────


def extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text.strip()


def normalize_label(text: str) -> str:
    return str(text).strip().upper().rstrip(".):")


def parse_choice_label(prediction_text: str, choices: list[dict]) -> str:
    answer = extract_answer(prediction_text)
    label = normalize_label(answer)
    valid_labels = {normalize_label(choice.get("label", "")) for choice in choices}
    if label in valid_labels:
        return label

    answer_lower = answer.lower()
    for choice in choices:
        choice_label = normalize_label(choice.get("label", ""))
        choice_text = str(choice.get("text", "")).strip()
        if choice_text and choice_text.lower() == answer_lower:
            return choice_label

    first_token = normalize_label(answer.split()[0]) if answer.split() else ""
    if first_token in valid_labels:
        return first_token
    return label


def evaluate(prediction_text: str, correct_choice: dict, choices: list[dict]) -> dict:
    predicted_label = parse_choice_label(prediction_text, choices)
    correct_label = normalize_label(correct_choice.get("label", ""))
    predicted_text = ""
    correct_text = str(correct_choice.get("text", "")).strip()
    for choice in choices:
        if normalize_label(choice.get("label", "")) == predicted_label:
            predicted_text = str(choice.get("text", "")).strip()
            break
    is_correct = float(predicted_label == correct_label)
    return {
        "em": is_correct,
        "f1": is_correct,
        "sub_em": is_correct,
        "predicted_answer": predicted_label or extract_answer(prediction_text),
        "predicted_label": predicted_label,
        "predicted_text": predicted_text,
        "correct_label": correct_label,
        "correct_text": correct_text,
    }
